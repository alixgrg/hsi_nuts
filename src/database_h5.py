# src/database_h5.py

from pathlib import Path
import json
import numpy as np
import h5py


ATTR_NONE = "__H5DB_NONE__"
ATTR_JSON_PREFIX = "__H5DB_JSON__:"

IMAGE_ARRAY_KEYS = {
    "cube",
    "image_ref",
    "mask",
    "labels",
    "wavelengths",
}

OBJECT_ARRAY_KEYS_COMPACT = {
    "mask",
    "positions_global",
    "positions_local",
    "spectra",
    "mean_spectrum",
    "std_spectrum",
    "median_spectrum",
    "wavelengths",
}

OBJECT_HEAVY_KEYS = {
    "mask_global",
    "cube_crop",
    "image_ref_crop",
}


def _json_default(value):
    """Convert numpy objects to JSON-compatible Python objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _to_h5_attr(value):
    """Convert Python value to a safe HDF5 attribute."""
    if value is None:
        return ATTR_NONE
    if isinstance(value, (dict, list, tuple)):
        return ATTR_JSON_PREFIX + json.dumps(value, default=_json_default)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    # Fallback for unusual metadata values
    return ATTR_JSON_PREFIX + json.dumps(value, default=_json_default)


def _from_h5_attr(value):
    """Convert HDF5 attribute back to Python value."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if value == ATTR_NONE:
            return None
        if value.startswith(ATTR_JSON_PREFIX):
            return json.loads(value[len(ATTR_JSON_PREFIX):])
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_attrs(group, record, skip_keys):
    """Write non-array metadata as HDF5 attributes."""
    for key, value in record.items():
        if key in skip_keys:
            continue
        if isinstance(value, np.ndarray):
            continue
        try:
            group.attrs[key] = _to_h5_attr(value)
        except Exception:
            group.attrs[key] = _to_h5_attr(str(value))


def _read_attrs(group):
    """Read HDF5 attributes into a Python dict."""
    out = {}
    for key, value in group.attrs.items():
        out[key] = _from_h5_attr(value)
    # Restore some frequent tuple-like fields
    for tuple_key in ("bbox", "centroid"):
        if tuple_key in out and isinstance(out[tuple_key], list):
            out[tuple_key] = tuple(out[tuple_key])
    return out


def _write_dataset(group, name, array, compression="gzip", compression_opts=4):
    """Write an array dataset, using compression for non-trivial arrays."""
    if array is None:
        return
    arr = np.asarray(array)
    if arr.dtype == object:
        return
    kwargs = {}
    if arr.size > 1000:
        kwargs = {
            "compression": compression,
            "compression_opts": compression_opts,
            "shuffle": True,
        }
    group.create_dataset(name, data=arr, **kwargs)


def save_nir_uco_h5(
    object_database,
    image_database,
    path,
    include_heavy_object_arrays=False,
    compression="gzip",
    compression_opts=4,
):
    """
    Save NIR UCO image_database and object_database to one HDF5 file.

    Parameters
    ----------
    object_database : dict
        Object-level database returned by build_minimal_nir_uco_object_database.
    image_database : dict
        Image-level database returned by build_minimal_nir_uco_object_database.
    path : str or Path
        Output HDF5 path.
    include_heavy_object_arrays : bool
        If False, do not save mask_global, cube_crop and image_ref_crop for each object.
        They can be reconstructed at loading time from image data + object bbox + object mask.
        If True, the file is larger but closer to the original Python dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    object_array_keys = set(OBJECT_ARRAY_KEYS_COMPACT)
    if include_heavy_object_arrays:
        object_array_keys |= OBJECT_HEAVY_KEYS

    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = "nir_uco_object_database"
        h5.attrs["version"] = "1.0"
        h5.attrs["n_images"] = len(image_database)
        h5.attrs["n_objects"] = len(object_database)
        h5.attrs["include_heavy_object_arrays"] = bool(include_heavy_object_arrays)
        images_group = h5.create_group("images")
        objects_group = h5.create_group("objects")
        
        # Image records
        for image_key, image_record in image_database.items():
            g = images_group.create_group(str(image_key))
            for array_key in IMAGE_ARRAY_KEYS:
                if array_key in image_record and image_record[array_key] is not None:
                    _write_dataset(
                        g,
                        array_key,
                        image_record[array_key],
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            _write_attrs(g, image_record, skip_keys=IMAGE_ARRAY_KEYS)

        # Object records
        for object_id, object_record in object_database.items():
            g = objects_group.create_group(str(object_id))
            for array_key in object_array_keys:
                if array_key in object_record and object_record[array_key] is not None:
                    _write_dataset(
                        g,
                        array_key,
                        object_record[array_key],
                        compression=compression,
                        compression_opts=compression_opts,
                    )
            _write_attrs(g, object_record, skip_keys=object_array_keys)

    return path


def _read_group_record(group):
    """Read one HDF5 group into a Python dict."""
    record = _read_attrs(group)
    for key, dataset in group.items():
        record[key] = dataset[()]
    return record


def _reconstruct_heavy_object_arrays(object_database, image_database):
    """
    Reconstruct mask_global, cube_crop and image_ref_crop if they were not saved.

    This keeps the HDF5 file smaller while preserving compatibility with code
    that expects these keys.
    """
    for _, obj in object_database.items():
        source_key = obj.get("source_clean_key")
        if source_key not in image_database:
            continue
        img = image_database[source_key]
        if "bbox" not in obj or "mask" not in obj:
            continue
        min_row, min_col, max_row, max_col = obj["bbox"]
        mask_crop = obj["mask"].astype(bool)
        if "labels" in img and "mask_global" not in obj:
            shape = img["labels"].shape
            mask_global = np.zeros(shape, dtype=bool)
            mask_global[min_row:max_row, min_col:max_col] = mask_crop
            obj["mask_global"] = mask_global
        if "cube" in img and "cube_crop" not in obj:
            obj["cube_crop"] = img["cube"][min_row:max_row, min_col:max_col, :]
        if "image_ref" in img and "image_ref_crop" not in obj:
            obj["image_ref_crop"] = img["image_ref"][min_row:max_row, min_col:max_col]

    return object_database


def load_nir_uco_h5(path, reconstruct_heavy_object_arrays=True):
    """
    Load a NIR UCO HDF5 database.

    Returns
    -------
    object_database : dict
    image_database : dict
    """
    path = Path(path)
    object_database = {}
    image_database = {}

    with h5py.File(path, "r") as h5:
        for image_key, group in h5["images"].items():
            image_database[image_key] = _read_group_record(group)
        for object_id, group in h5["objects"].items():
            object_database[object_id] = _read_group_record(group)

    if reconstruct_heavy_object_arrays:
        object_database = _reconstruct_heavy_object_arrays(
            object_database,
            image_database,
        )

    return object_database, image_database