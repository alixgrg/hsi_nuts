# src/database_h5.py

from pathlib import Path
import json
import os
import tempfile
import numpy as np
import h5py
import pandas as pd

from src.protocol_governance import (
    sha256_file,
    sha256_ndarray,
    sha256_payload,
)

ATTR_NONE = "__H5DB_NONE__"
ATTR_JSON_PREFIX = "__H5DB_JSON__:"
H5_FORMAT = "nir_uco_object_database"
H5_VERSION = "1.0"

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


def _logical_record_payload(record, array_keys):
    payload = {}
    for key in sorted(record):
        value = record[key]
        if key in array_keys:
            if value is not None:
                payload[key] = {
                    "array_sha256": sha256_ndarray(value),
                }
        elif isinstance(value, np.ndarray):
            continue
        else:
            payload[key] = value
    return payload


def database_content_hash(object_database, image_database) -> str:
    """Hash the compact logical database independently of HDF5 layout."""
    payload = {
        "format": H5_FORMAT,
        "version": H5_VERSION,
        "images": {
            str(record_id): _logical_record_payload(
                record,
                IMAGE_ARRAY_KEYS,
            )
            for record_id, record in sorted(
                image_database.items(),
                key=lambda pair: str(pair[0]),
            )
        },
        "objects": {
            str(record_id): _logical_record_payload(
                record,
                OBJECT_ARRAY_KEYS_COMPACT,
            )
            for record_id, record in sorted(
                object_database.items(),
                key=lambda pair: str(pair[0]),
            )
        },
    }
    return sha256_payload(payload)


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
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _to_h5_attr(value):
    """Convert Python value to a safe HDF5 attribute."""
    if value is None:
        return ATTR_NONE
    if isinstance(value, (dict, list, tuple, set)):
        return ATTR_JSON_PREFIX + json.dumps(value, default=_json_default, sort_keys=True)
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
        raise TypeError(
            f"Cannot write object-dtype array dataset {name!r} to HDF5 without loss."
        )
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
    raw_file_sha256=None,
    protocol_version=None,
    configuration_hash=None,
    content_hash=None,
    atomic=True,
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
    expected_content_hash = (
        database_content_hash(object_database, image_database)
        if content_hash is None
        else str(content_hash)
    )

    object_array_keys = set(OBJECT_ARRAY_KEYS_COMPACT)
    if include_heavy_object_arrays:
        object_array_keys |= OBJECT_HEAVY_KEYS

    if atomic:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        )
        handle.close()
        write_path = Path(handle.name)
    else:
        write_path = path

    try:
        with h5py.File(write_path, "w") as h5:
            h5.attrs["format"] = H5_FORMAT
            h5.attrs["version"] = H5_VERSION
            h5.attrs["n_images"] = len(image_database)
            h5.attrs["n_objects"] = len(object_database)
            h5.attrs["include_heavy_object_arrays"] = bool(include_heavy_object_arrays)
            h5.attrs["raw_file_sha256"] = _to_h5_attr(raw_file_sha256)
            h5.attrs["protocol_version"] = _to_h5_attr(protocol_version)
            h5.attrs["configuration_hash"] = _to_h5_attr(configuration_hash)
            h5.attrs["database_content_sha256"] = expected_content_hash
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

        validate_nir_uco_h5(write_path, deep=True)
        reloaded_objects, reloaded_images = load_nir_uco_h5(
            write_path,
            reconstruct_heavy_object_arrays=False,
        )
        observed_content_hash = database_content_hash(
            reloaded_objects,
            reloaded_images,
        )
        if observed_content_hash != expected_content_hash:
            raise ValueError(
                "Logical database hash changed after HDF5 roundtrip: "
                f"expected={expected_content_hash}, "
                f"observed={observed_content_hash}"
            )
        if atomic:
            os.replace(write_path, path)
    except Exception:
        if atomic and write_path.exists():
            write_path.unlink()
        raise

    return path


def _validate_open_h5(h5):
    file_format = _from_h5_attr(h5.attrs.get("format"))
    if file_format != H5_FORMAT:
        raise ValueError(f"Unsupported HDF5 format: {file_format!r}")
    file_version = _from_h5_attr(h5.attrs.get("version"))
    if file_version != H5_VERSION:
        raise ValueError(
            f"Unsupported HDF5 schema version: {file_version!r}; "
            f"expected {H5_VERSION!r}."
        )
    missing_groups = [name for name in ("images", "objects") if name not in h5]
    if missing_groups:
        raise ValueError(f"NIR UCO HDF5 file is missing group(s): {missing_groups}")
    n_images = int(_from_h5_attr(h5.attrs.get("n_images", 0)))
    n_objects = int(_from_h5_attr(h5.attrs.get("n_objects", 0)))
    if n_images != len(h5["images"]):
        raise ValueError(f"HDF5 image count mismatch: attr={n_images}, group={len(h5['images'])}")
    if n_objects != len(h5["objects"]):
        raise ValueError(f"HDF5 object count mismatch: attr={n_objects}, group={len(h5['objects'])}")


def _validation_row(scope, record_id, check, passed, detail=""):
    return {
        "scope": str(scope),
        "record_id": str(record_id),
        "check": str(check),
        "passed": bool(passed),
        "detail": str(detail),
    }


def _arrays_match(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        return bool(np.allclose(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _deep_h5_rows(h5):
    rows = []
    required_image_datasets = ("cube", "image_ref", "mask", "labels", "wavelengths")
    required_object_datasets = (
        "spectra",
        "positions_global",
        "mean_spectrum",
        "median_spectrum",
        "std_spectrum",
        "wavelengths",
    )

    for image_id, group in h5["images"].items():
        missing = [name for name in required_image_datasets if name not in group]
        rows.append(
            _validation_row(
                "image",
                image_id,
                "required_datasets",
                not missing,
                f"missing={missing}" if missing else "",
            )
        )
        if missing:
            continue

        cube = np.asarray(group["cube"])
        image_ref = np.asarray(group["image_ref"])
        mask = np.asarray(group["mask"])
        labels = np.asarray(group["labels"])
        wavelengths = np.asarray(group["wavelengths"])
        spatial_shape = cube.shape[:2] if cube.ndim == 3 else None

        rows.extend(
            [
                _validation_row("image", image_id, "cube_is_3d", cube.ndim == 3, cube.shape),
                _validation_row(
                    "image",
                    image_id,
                    "spatial_shapes_match",
                    (
                        spatial_shape is not None
                        and image_ref.shape == spatial_shape
                        and mask.shape == spatial_shape
                        and labels.shape == spatial_shape
                    ),
                    (
                        f"cube={cube.shape}, image_ref={image_ref.shape}, "
                        f"mask={mask.shape}, labels={labels.shape}"
                    ),
                ),
                _validation_row(
                    "image",
                    image_id,
                    "wavelength_length",
                    cube.ndim == 3
                    and wavelengths.ndim == 1
                    and len(wavelengths) == cube.shape[2],
                    f"cube_bands={cube.shape[2] if cube.ndim == 3 else None}, axis={len(wavelengths)}",
                ),
            ]
        )

    for object_id, group in h5["objects"].items():
        missing = [name for name in required_object_datasets if name not in group]
        rows.append(
            _validation_row(
                "object",
                object_id,
                "required_datasets",
                not missing,
                f"missing={missing}" if missing else "",
            )
        )
        if missing:
            continue

        spectra = np.asarray(group["spectra"])
        positions = np.asarray(group["positions_global"])
        mean = np.asarray(group["mean_spectrum"])
        median = np.asarray(group["median_spectrum"])
        std = np.asarray(group["std_spectrum"])
        wavelengths = np.asarray(group["wavelengths"])
        n_pixels = int(_from_h5_attr(group.attrs.get("n_pixels", -1)))
        n_bands = int(_from_h5_attr(group.attrs.get("n_bands", -1)))
        area_pixels = int(_from_h5_attr(group.attrs.get("area_pixels", -1)))

        rows.extend(
            [
                _validation_row(
                    "object",
                    object_id,
                    "spectra_shape",
                    spectra.ndim == 2
                    and spectra.shape == (n_pixels, n_bands)
                    and n_pixels == area_pixels,
                    (
                        f"spectra={spectra.shape}, n_pixels={n_pixels}, "
                        f"area_pixels={area_pixels}, n_bands={n_bands}"
                    ),
                ),
                _validation_row(
                    "object",
                    object_id,
                    "positions_shape",
                    positions.ndim == 2
                    and positions.shape == (n_pixels, 2),
                    f"positions={positions.shape}, n_pixels={n_pixels}",
                ),
                _validation_row(
                    "object",
                    object_id,
                    "summary_spectrum_lengths",
                    all(len(arr) == n_bands for arr in (mean, median, std, wavelengths)),
                    (
                        f"mean={len(mean)}, median={len(median)}, std={len(std)}, "
                        f"axis={len(wavelengths)}, n_bands={n_bands}"
                    ),
                ),
            ]
        )
    return rows


def _expected_database_rows(
    path,
    *,
    expected_image_db=None,
    expected_object_db=None,
):
    rows = []
    loaded_object_db, loaded_image_db = load_nir_uco_h5(
        path,
        reconstruct_heavy_object_arrays=True,
    )

    if expected_image_db is not None:
        expected_keys = set(map(str, expected_image_db))
        loaded_keys = set(map(str, loaded_image_db))
        rows.append(
            _validation_row(
                "database",
                "images",
                "record_ids",
                expected_keys == loaded_keys,
                (
                    f"missing={sorted(expected_keys - loaded_keys)}, "
                    f"unexpected={sorted(loaded_keys - expected_keys)}"
                ),
            )
        )
        for image_id in sorted(expected_keys & loaded_keys):
            expected = expected_image_db[image_id]
            loaded = loaded_image_db[image_id]
            rows.append(
                _validation_row(
                    "image",
                    image_id,
                    "field_names",
                    set(expected) == set(loaded),
                    (
                        f"missing={sorted(set(expected) - set(loaded))}, "
                        f"unexpected={sorted(set(loaded) - set(expected))}"
                    ),
                )
            )
            for field in ("cube", "labels", "mask", "wavelengths"):
                if field in expected and field in loaded:
                    rows.append(
                        _validation_row(
                            "image",
                            image_id,
                            f"memory_roundtrip_{field}",
                            _arrays_match(expected[field], loaded[field]),
                            f"memory={np.asarray(expected[field]).shape}, h5={np.asarray(loaded[field]).shape}",
                        )
                    )

    if expected_object_db is not None:
        expected_keys = set(map(str, expected_object_db))
        loaded_keys = set(map(str, loaded_object_db))
        rows.append(
            _validation_row(
                "database",
                "objects",
                "record_ids",
                expected_keys == loaded_keys,
                (
                    f"missing={sorted(expected_keys - loaded_keys)}, "
                    f"unexpected={sorted(loaded_keys - expected_keys)}"
                ),
            )
        )
        for object_id in sorted(expected_keys & loaded_keys):
            expected = expected_object_db[object_id]
            loaded = loaded_object_db[object_id]
            rows.append(
                _validation_row(
                    "object",
                    object_id,
                    "field_names",
                    set(expected) == set(loaded),
                    (
                        f"missing={sorted(set(expected) - set(loaded))}, "
                        f"unexpected={sorted(set(loaded) - set(expected))}"
                    ),
                )
            )
            for field in ("object_id", "n_pixels", "n_bands", "area_pixels"):
                rows.append(
                    _validation_row(
                        "object",
                        object_id,
                        f"memory_roundtrip_{field}",
                        expected.get(field) == loaded.get(field),
                        f"memory={expected.get(field)!r}, h5={loaded.get(field)!r}",
                    )
                )
            for field in (
                "spectra",
                "positions_global",
                "mean_spectrum",
                "median_spectrum",
                "std_spectrum",
                "wavelengths",
            ):
                if field in expected and field in loaded:
                    rows.append(
                        _validation_row(
                            "object",
                            object_id,
                            f"memory_roundtrip_{field}",
                            _arrays_match(expected[field], loaded[field]),
                            f"memory={np.asarray(expected[field]).shape}, h5={np.asarray(loaded[field]).shape}",
                        )
                    )
    return rows


def validate_nir_uco_h5(
    path,
    *,
    expected_image_db=None,
    expected_object_db=None,
    deep=False,
    return_report=False,
):
    """Validate HDF5 structure and optionally perform a deep memory roundtrip.

    With ``return_report=True`` the function returns a compact dataframe and
    leaves failure handling to the caller. Otherwise any failed check raises
    ``ValueError`` and a valid file returns ``True``.
    """
    path = Path(path)
    rows = []
    stored_content_hash = None
    try:
        with h5py.File(path, "r") as h5:
            _validate_open_h5(h5)
            rows.append(_validation_row("file", path.name, "basic_structure", True))
            if deep:
                rows.extend(_deep_h5_rows(h5))
                stored_content_hash = _from_h5_attr(
                    h5.attrs.get("database_content_sha256")
                )
    except Exception as exc:
        rows.append(
            _validation_row(
                "file",
                path.name,
                "basic_structure",
                False,
                repr(exc),
            )
        )

    if rows and rows[0]["passed"] and deep:
        try:
            objects, images = load_nir_uco_h5(
                path,
                reconstruct_heavy_object_arrays=False,
            )
            observed = database_content_hash(objects, images)
            rows.append(
                _validation_row(
                    "file",
                    path.name,
                    "logical_content_hash",
                    bool(stored_content_hash)
                    and observed == stored_content_hash,
                    (
                        f"stored={stored_content_hash}, "
                        f"observed={observed}"
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                _validation_row(
                    "file",
                    path.name,
                    "logical_content_hash",
                    False,
                    repr(exc),
                )
            )

    if rows and rows[0]["passed"] and (
        expected_image_db is not None or expected_object_db is not None
    ):
        try:
            rows.extend(
                _expected_database_rows(
                    path,
                    expected_image_db=expected_image_db,
                    expected_object_db=expected_object_db,
                )
            )
        except Exception as exc:
            rows.append(
                _validation_row(
                    "database",
                    path.name,
                    "memory_roundtrip",
                    False,
                    repr(exc),
                )
            )

    report = pd.DataFrame(
        rows,
        columns=["scope", "record_id", "check", "passed", "detail"],
    )
    if return_report:
        return report

    failed = report.loc[~report["passed"]]
    if not failed.empty:
        first = failed.head(5).to_dict(orient="records")
        raise ValueError(f"NIR UCO HDF5 validation failed: {first}")
    return True


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


def load_nir_uco_h5(
    path,
    reconstruct_heavy_object_arrays=True,
    batches=None,
):
    """
    Load a NIR UCO HDF5 database.

    When ``batches`` is provided, records are filtered from their HDF5
    metadata before their datasets are read. Records without a scalar batch
    (for example cross-batch mixtures) are excluded from that restricted load.

    Returns
    -------
    object_database : dict
    image_database : dict
    """
    path = Path(path)
    object_database = {}
    image_database = {}
    allowed_batches = (
        None
        if batches is None
        else {int(batch) for batch in batches}
    )

    def batch_is_allowed(group):
        if allowed_batches is None:
            return True
        batch = _from_h5_attr(group.attrs.get("batch"))
        if batch is None:
            return False
        try:
            return int(batch) in allowed_batches
        except (TypeError, ValueError):
            return False

    with h5py.File(path, "r") as h5:
        _validate_open_h5(h5)
        for image_key, group in h5["images"].items():
            if batch_is_allowed(group):
                image_database[image_key] = _read_group_record(group)
        for object_id, group in h5["objects"].items():
            if batch_is_allowed(group):
                object_database[object_id] = _read_group_record(group)

    if reconstruct_heavy_object_arrays:
        object_database = _reconstruct_heavy_object_arrays(
            object_database,
            image_database,
        )

    return object_database, image_database


def build_database_manifest(
    object_database,
    image_database,
    h5_path,
    *,
    database_id,
    wavelength_mode,
    data_mode,
    protocol_version,
    validation_report=None,
) -> pd.DataFrame:
    """Build the single-row compact persistence manifest."""
    axes = [
        np.asarray(record.get("wavelengths"), dtype=float)
        for record in image_database.values()
        if record.get("wavelengths") is not None
    ]
    axis = axes[0] if axes else np.array([], dtype=float)
    if validation_report is None:
        validation_report = validate_nir_uco_h5(
            h5_path,
            deep=True,
            return_report=True,
        )
    failures = validation_report.loc[~validation_report["passed"]]
    row = {
        "database_id": str(database_id),
        "wavelength_mode": str(wavelength_mode),
        "data_mode": str(data_mode),
        "n_images": int(len(image_database)),
        "n_objects": int(len(object_database)),
        "n_bands": int(len(axis)),
        "wavelength_min_nm": (
            float(np.min(axis)) if axis.size else np.nan
        ),
        "wavelength_max_nm": (
            float(np.max(axis)) if axis.size else np.nan
        ),
        "hdf5_valid": bool(failures.empty),
        "validation_failures": int(len(failures)),
        "h5_schema_version": H5_VERSION,
        "protocol_version": str(protocol_version),
        "database_content_sha256": database_content_hash(
            object_database,
            image_database,
        ),
        "h5_file_sha256": sha256_file(h5_path),
    }
    return pd.DataFrame([row])
