from pathlib import Path
import numpy as np
import scipy.io
import h5py

from src.protocol_governance import sha256_file


def _matlab_version_from_header(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"\x89HDF"):
        return "7.3"
    text = header.decode("latin-1", errors="ignore")
    marker = "MATLAB "
    if marker in text and " MAT-file" in text:
        return text.split(marker, 1)[1].split(" MAT-file", 1)[0].strip()
    return "unknown"


def load_mat_file(path, *, return_metadata: bool = False):
    """Load a MATLAB file and optionally return auditable load metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    file_sha256 = sha256_file(path)
    matlab_version = _matlab_version_from_header(path)

    try:
        data = scipy.io.loadmat(path)
        data = {
            k: v for k, v in data.items()
            if not k.startswith("__")
        }
        backend = "scipy.io.loadmat"
    except (NotImplementedError, ValueError):
        data = {}
        with h5py.File(path, "r") as f:
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = np.array(obj)
            f.visititems(visitor)
        backend = "h5py"
        matlab_version = "7.3"

    if not return_metadata:
        return data
    return data, {
        "path": str(path.resolve()),
        "backend": backend,
        "matlab_version": matlab_version,
        "file_sha256": file_sha256,
        "n_entries": int(len(data)),
    }
