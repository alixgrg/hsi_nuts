import hashlib

import numpy as np

from src.utils import filter_records


def stable_object_seed(global_seed: int, object_id: str) -> int:
    """Derive a process-independent NumPy seed for one object."""
    digest = hashlib.sha256(
        f"{int(global_seed)}\0{str(object_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def pixel_selection_hash(indices) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def object_db_to_matrix(
    object_db,
    level="object",
    spectrum_field="mean_spectrum",
    filters=None,
    m=40,
    random_state=42,
    replace=False,
    balanced_pixel_strategy="random",
    under_m_policy=None,
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
    under_m_policy : {"short", "exclude", "replace", "error"} or None
        Explicit behavior for objects with fewer than ``m`` pixels. ``None``
        preserves compatibility: ``replace=True`` selects ``"replace"`` and
        otherwise selects ``"short"``.

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
            under_m_policy=under_m_policy,
        )

    raise ValueError(
        "level must be one of: 'object', 'pixel', 'balanced_pixel'."
    )


def _common_object_metadata(obj_id, obj):
    return {
        "object_id": obj_id,
        "source_image": obj.get("source_clean_key"),
        "batch": obj.get("batch"),
        "label": obj.get("object_nut_type"),
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
    under_m_policy=None,
):
    X_list = []
    rows = []

    for obj_id, obj in objects:
        spectra = np.asarray(obj["spectra"], dtype=float)
        positions = np.asarray(obj["positions_global"])
        
        idx = select_balanced_pixel_indices(
            obj,
            m=m,
            random_state=random_state,
            object_id=str(obj_id),
            replace=replace,
            balanced_pixel_strategy=balanced_pixel_strategy,
            under_m_policy=under_m_policy,
        )
        if idx is None or len(idx) == 0:
            continue

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


def select_balanced_pixel_indices(
    obj,
    m=40,
    random_state=42,
    *,
    rng=None,
    object_id=None,
    return_diagnostics=False,
    replace=False,
    balanced_pixel_strategy="random",
    under_m_policy=None,
):
    spectra = np.asarray(obj["spectra"], dtype=float)
    positions = np.asarray(obj["positions_global"], dtype=float)
    n_pixels = spectra.shape[0]
    m = int(m)
    if m <= 0:
        raise ValueError("m must be a strictly positive integer.")
    policy = (
        ("replace" if replace else "short")
        if under_m_policy is None
        else str(under_m_policy)
    )
    valid_policies = {"short", "exclude", "replace", "error"}
    if policy not in valid_policies:
        raise ValueError(
            f"under_m_policy must be one of {sorted(valid_policies)}, got {policy!r}."
        )

    if rng is None:
        seed = (
            int(random_state)
            if object_id is None
            else stable_object_seed(random_state, str(object_id))
        )
        rng = np.random.default_rng(seed)
    else:
        seed = None

    def finish(indices, status):
        if not return_diagnostics:
            return indices
        values = (
            np.array([], dtype=int)
            if indices is None
            else np.asarray(indices, dtype=int)
        )
        return indices, {
            "object_id": None if object_id is None else str(object_id),
            "n_available": int(n_pixels),
            "n_selected": int(values.size),
            "selection_hash": pixel_selection_hash(values),
            "object_seed": seed,
            "status": str(status),
        }

    if n_pixels == 0:
        if policy == "exclude":
            return finish(None, "excluded_under_m")
        if policy == "error":
            raise ValueError("Cannot sample pixels from an empty object.")
        return finish(np.array([], dtype=int), "empty")

    if balanced_pixel_strategy == "random":
        if n_pixels >= m:
            return finish(
                rng.choice(n_pixels, size=m, replace=False),
                "accepted",
            )
        if policy == "replace":
            return finish(
                rng.choice(n_pixels, size=m, replace=True),
                "accepted_with_replacement",
            )
        if policy == "short":
            return finish(np.arange(n_pixels), "accepted_short")
        if policy == "exclude":
            return finish(None, "excluded_under_m")
        raise ValueError(f"Object has {n_pixels} pixels, fewer than requested m={m}.")
    if balanced_pixel_strategy in {"center", "center_closest"}:
        centroid = np.asarray(obj.get("centroid", positions.mean(axis=0)), dtype=float)
        distances = np.linalg.norm(positions - centroid[None, :], axis=1)
        order = np.argsort(distances)
        if n_pixels >= m:
            return finish(order[:m], "accepted")
        if policy == "replace":
            extra = rng.choice(order, size=m - n_pixels, replace=True)
            return finish(
                np.concatenate([order, extra]),
                "accepted_with_replacement",
            )
        if policy == "short":
            return finish(order, "accepted_short")
        if policy == "exclude":
            return finish(None, "excluded_under_m")
        raise ValueError(f"Object has {n_pixels} pixels, fewer than requested m={m}.")

    raise ValueError(
        "balanced_pixel_strategy must be 'random' or 'center'."
    )


# Backward-compatible private name used by older notebooks.
_select_balanced_pixel_indices = select_balanced_pixel_indices
