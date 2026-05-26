from pathlib import Path
import numpy as np
import scipy.io
import h5py


def load_mat_file(path):

    path = Path(path)

    try:
        data = scipy.io.loadmat(path)
        data = {
            k: v for k, v in data.items()
            if not k.startswith("__")
        }
        return data

    except NotImplementedError:
        data = {}
        with h5py.File(path, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = np.array(obj)
            f.visititems(visitor)
        return data