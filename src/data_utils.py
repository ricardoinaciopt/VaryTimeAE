import os
import numpy as np
import pickle
import yaml

SCALER_FNAME = "scaler.pkl"


def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def load_data_with_mask(data_dir: str, dataset: str):
    loaded = np.load(os.path.join(data_dir, f"{dataset}.npz"))
    data = loaded["data"]
    mask = loaded["mask"] if "mask" in loaded.files else None
    return data, mask


def save_data(data: np.ndarray, output_file: str) -> None:
    """
    Save data to a .npz file.

    Args:
        data (np.ndarray): The data to save.
        output_file (str): The path to the .npz file to save the data to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, data=data)


def split_data_with_mask(
    data, mask, valid_perc: float, shuffle: bool = True, seed: int = 123
):
    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))

    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    data = data[idx]
    mask = mask[idx] if mask is not None else None

    train_idx = idx[:N_train]
    valid_idx = idx[N_train:]

    train_data = data[:N_train]
    valid_data = data[N_train:]
    train_mask = mask[:N_train] if mask is not None else None
    valid_mask = mask[N_train:] if mask is not None else None

    return train_data, valid_data, train_mask, valid_mask, train_idx, valid_idx


def get_npz_data(input_file: str) -> np.ndarray:
    """
    Load data from a .npz file.

    Args:
        input_file (str): The path to the .npz file.

    Returns:
        np.ndarray: The data array extracted from the .npz file.
    """
    loaded = np.load(input_file)
    return loaded["data"]


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def inverse_transform_data(
    data: np.ndarray, scaler, mask: np.ndarray | None = None
) -> np.ndarray:
    N, T, F = data.shape
    d2 = data.reshape(-1, F).copy()

    if mask is None:
        inv = scaler.inverse_transform(d2).reshape(N, T, F)
        return inv

    m = mask
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    m_flat = m.reshape(-1) > 0

    out = np.zeros_like(d2, dtype=np.float32)
    out[m_flat] = scaler.inverse_transform(d2[m_flat]).astype(np.float32)
    return out.reshape(N, T, F)


def scale_data(train_data, valid_data):
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    return scaled_train_data, scaled_valid_data, scaler


from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_data_with_mask(train_data, valid_data, train_mask, valid_mask):
    Ntr, T, F = train_data.shape
    Nva = valid_data.shape[0]

    scaler = MinMaxScaler()

    tr2 = train_data.reshape(-1, F)
    va2 = valid_data.reshape(-1, F)

    trm = train_mask.reshape(-1) > 0
    vam = valid_mask.reshape(-1) > 0

    scaler.fit(tr2[trm])

    tr_scaled = np.zeros_like(tr2, dtype=np.float32)
    va_scaled = np.zeros_like(va2, dtype=np.float32)

    tr_scaled[trm] = scaler.transform(tr2[trm]).astype(np.float32)
    va_scaled[vam] = scaler.transform(va2[vam]).astype(np.float32)

    return tr_scaled.reshape(Ntr, T, F), va_scaled.reshape(Nva, T, F), scaler


def save_scaler(scaler: MinMaxScaler, dir_path: str) -> None:
    """
    Save a MinMaxScaler to a file.

    Args:
        scaler (MinMaxScaler): The scaler to save.
        dir_path (str): The path to the directory where the scaler will be saved.

    Returns:
        None
    """
    os.makedirs(dir_path, exist_ok=True)
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "wb") as file:
        pickle.dump(scaler, file)


def load_scaler(dir_path: str) -> MinMaxScaler:
    """
    Load a MinMaxScaler from a file.

    Args:
        dir_path (str): The path to the file from which the scaler will be loaded.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "rb") as file:
        scaler = pickle.load(file)
    return scaler
