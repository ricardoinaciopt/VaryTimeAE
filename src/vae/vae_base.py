import os, warnings, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Mean
from tensorflow.keras.backend import random_normal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(Model, ABC):
    model_name = None

    def __init__(
        self,
        seq_len,
        feat_dim,
        latent_dim,
        reconstruction_wt=3.0,
        batch_size=16,
        **kwargs,
    ):
        super(BaseVariationalAutoencoder, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.batch_size = batch_size
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.encoder = None
        self.decoder = None

    def fit_on_data(self, train_data, max_epochs=1000, verbose=0, train_mask=None):
        loss_to_monitor = "total_loss"
        early_stopping = EarlyStopping(
            monitor=loss_to_monitor, min_delta=1e-2, patience=50, mode="min"
        )
        reduce_lr = ReduceLROnPlateau(
            monitor=loss_to_monitor, factor=0.5, patience=30, mode="min"
        )

        sample_weight = None
        if train_mask is not None:
            sample_weight = train_mask
            if sample_weight.ndim == 3 and sample_weight.shape[-1] == 1:
                sample_weight = sample_weight[..., 0]  # (N, T)

        self.fit(
            train_data,
            epochs=max_epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
            sample_weight=sample_weight,
        )

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded

    def get_num_trainable_variables(self):
        trainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.trainable_weights])
        )
        nonTrainableParams = int(
            np.sum([np.prod(v.get_shape()) for v in self.non_trainable_weights])
        )
        totalParams = trainableParams + nonTrainableParams
        return trainableParams, nonTrainableParams, totalParams

    def get_prior_samples(self, num_samples):
        Z = np.random.randn(num_samples, self.latent_dim)
        samples = self.decoder.predict(Z, verbose=0)
        return samples

    def get_prior_samples_given_Z(self, Z):
        samples = self.decoder.predict(Z)
        return samples

    @abstractmethod
    def _get_encoder(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_decoder(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def _get_reconstruction_loss(self, X, X_recons, sample_weight=None):
        err = tf.math.squared_difference(X, X_recons)  # (B,T,F)

        if sample_weight is None:
            per_sample = tf.reduce_mean(err, axis=[1, 2])  # (B,)
            return tf.reduce_mean(per_sample)

        M = tf.cast(sample_weight, err.dtype)  # (B,T)
        M = M[..., None]  # (B,T,1)
        num = tf.reduce_sum(err * M, axis=[1, 2])  # (B,)
        den = tf.reduce_sum(M, axis=[1, 2]) + 1e-8  # (B,)
        per_sample = num / den
        return tf.reduce_mean(per_sample)

    def train_step(self, data):
        X, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        if sample_weight is not None:
            X = X * tf.cast(sample_weight[..., None], X.dtype)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X, training=True)

            z = z_mean

            reconstruction = self.decoder(z, training=True)

            reconstruction_loss = self._get_reconstruction_loss(
                X, reconstruction, sample_weight
            )

            kl_loss = tf.constant(0.0, dtype=reconstruction_loss.dtype)

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        X, _, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        if sample_weight is not None:
            X = X * tf.cast(sample_weight[..., None], X.dtype)

        z_mean, z_log_var, z = self.encoder(X, training=False)
        z = z_mean
        reconstruction = self.decoder(z, training=False)

        reconstruction_loss = self._get_reconstruction_loss(
            X, reconstruction, sample_weight
        )
        kl_loss = tf.constant(0.0, dtype=reconstruction_loss.dtype)
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_weights(self, model_dir):
        if self.model_name is None:
            raise ValueError("Model name not set.")
        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(
            encoder_wts, os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        joblib.dump(
            decoder_wts, os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

    def load_weights(self, model_dir):
        encoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_encoder_wts.h5")
        )
        decoder_wts = joblib.load(
            os.path.join(model_dir, f"{self.model_name}_decoder_wts.h5")
        )

        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.save_weights(model_dir)
        dict_params = {
            "seq_len": self.seq_len,
            "feat_dim": self.feat_dim,
            "latent_dim": self.latent_dim,
            "reconstruction_wt": self.reconstruction_wt,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
        }
        params_file = os.path.join(model_dir, f"{self.model_name}_parameters.pkl")
        joblib.dump(dict_params, params_file)


#####################################################################################################
#####################################################################################################


if __name__ == "__main__":
    pass
