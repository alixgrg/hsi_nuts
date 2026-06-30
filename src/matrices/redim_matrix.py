import numpy as np

from src.utils import filter_records


def object_db_to_matrix(
    object_db,
    level="object",
    spectrum_field="mean_spectrum",
    filters=None,
    m=40,
    random_state=42,
    replace=False,
    balanced_pixel_strategy="random",
):
    """
    Convert object database to a matrix.

    Parameters
    ----------
    object_db : dict
        Object-level database.
    level : {"object", "pixel", "balanced_pixel"}
        Observation level.
    spectrum_field : str
        Used only for level="object".
        Usually "mean_spectrum" or "median_spectrum".
    filters : dict or None
        Filters applied on object fields.
        Example:
            {
                "split": ["train_minimal"],
                "object_nut_type": ["almond", "peanut"],
                "source_clean_key": ["almond1", "peanut1"],
            }
    m : int
        Number of pixels sampled per object for level="balanced_pixel".
    random_state : int
        Random seed for balanced pixel sampling.
    replace : bool
        Sample with replacement if object has fewer than m pixels.
    balanced_pixel_strategy : {"random", "center", "center_closest"}
        balanced_pixel_strategy for selecting pixels.

    Returns
    -------
    X : ndarray, shape (N, B)
    y : ndarray, shape (N,)
    meta : dict
        Metadata arrays aligned with rows of X.
    """
    filters = filters or {}
    selected = filter_records(object_db, **filters)

    if level == "object":
        return _objects_to_object_matrix(
            selected,
            spectrum_field=spectrum_field,
        )

    if level == "pixel":
        return _objects_to_pixel_matrix(
            selected,
        )

    if level == "balanced_pixel":
        return _objects_to_balanced_pixel_matrix(
            selected,
            m=m,
            random_state=random_state,
            replace=replace,
            balanced_pixel_strategy=balanced_pixel_strategy,
        )

    raise ValueError(
        "level must be one of: 'object', 'pixel', 'balanced_pixel'."
    )


def _common_object_metadata(obj_id, obj):
    return {
        "object_id": obj_id,
        "label": obj.get("object_nut_type"),
        "source_image": obj.get("source_clean_key"),
        "batch": obj.get("batch"),
        "area": obj.get("area_pixels"),
        "sample_kind": obj.get("sample_kind"),
    }


def _objects_to_object_matrix(objects, spectrum_field="mean_spectrum"):
    X_list = []
    rows = []

    for obj_id, obj in objects:
        X_list.append(np.asarray(obj[spectrum_field], dtype=float))
        rows.append(_common_object_metadata(obj_id, obj))

    if not X_list:
        raise ValueError("No objects found with the requested filters.")

    X = np.vstack(X_list)
    meta = _metadata_rows_to_arrays(rows)
    y = meta["label"]

    return X, y, meta


def _objects_to_pixel_matrix(objects):
    X_list = []
    rows = []

    for obj_id, obj in objects:
        spectra = np.asarray(obj["spectra"], dtype=float)
        positions = np.asarray(obj["positions_global"])
        n_pixels = spectra.shape[0]

        X_list.append(spectra)

        base = _common_object_metadata(obj_id, obj)
        for i in range(n_pixels):
            row = base.copy()
            row["pixel_index"] = i
            row["row"] = positions[i, 0]
            row["col"] = positions[i, 1]
            rows.append(row)

    if not X_list:
        raise ValueError("No pixels found with the requested filters.")

    X = np.vstack(X_list)
    meta = _metadata_rows_to_arrays(rows)
    y = meta["label"]

    return X, y, meta


def _objects_to_balanced_pixel_matrix(
    objects,
    m=40,
    random_state=42,
    replace=False,
    balanced_pixel_strategy="random",
):
    rng = np.random.default_rng(random_state)

    X_list = []
    rows = []

    for obj_id, obj in objects:
        spectra = np.asarray(obj["spectra"], dtype=float)
        positions = np.asarray(obj["positions_global"])
        
        idx = _select_balanced_pixel_indices(obj, m=m, random_state=random_state, replace=replace, balanced_pixel_strategy=balanced_pixel_strategy)

        X_selected = spectra[idx]
        positions_selected = positions[idx]

        X_list.append(X_selected)

        base = _common_object_metadata(obj_id, obj)
        for local_i, pixel_i in enumerate(idx):
            row = base.copy()
            row["pixel_index"] = int(pixel_i)
            row["row"] = positions_selected[local_i, 0]
            row["col"] = positions_selected[local_i, 1]
            rows.append(row)

    if not X_list:
        raise ValueError("No pixels found with the requested filters.")

    X = np.vstack(X_list)
    meta = _metadata_rows_to_arrays(rows)
    y = meta["label"]

    return X, y, meta


def _metadata_rows_to_arrays(rows):
    keys = rows[0].keys()
    meta = {}

    for key in keys:
        meta[key] = np.asarray([row[key] for row in rows])

    return meta


def _select_balanced_pixel_indices(
    obj,
    m=40,
    random_state=42,
    replace=False,
    balanced_pixel_strategy="random",
):
    spectra = np.asarray(obj["spectra"], dtype=float)
    positions = np.asarray(obj["positions_global"], dtype=float)
    n_pixels = spectra.shape[0]

    if n_pixels == 0:
        return np.array([], dtype=int)

    rng = np.random.default_rng(random_state)
    if balanced_pixel_strategy == "random":
        if n_pixels >= m:
            return rng.choice(n_pixels, size=m, replace=False)
        if replace:
            return rng.choice(n_pixels, size=m, replace=True)
        return np.arange(n_pixels)
    if balanced_pixel_strategy in {"center", "center_closest"}:
        centroid = np.asarray(obj.get("centroid", positions.mean(axis=0)), dtype=float)
        distances = np.linalg.norm(positions - centroid[None, :], axis=1)
        order = np.argsort(distances)
        if n_pixels >= m:
            return order[:m]
        if replace:
            extra = rng.choice(order, size=m - n_pixels, replace=True)
            return np.concatenate([order, extra])

        return order

    raise ValueError(
        "balanced_pixel_strategy must be 'random' or 'center'."
    )