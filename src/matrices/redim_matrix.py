import hashlib

import numpy as np

from src.utils import filter_records
from src import experiment_config as expcfg
from src.spectra.band_selection import spectral_pixel_validity_report


_VALID_UNDER_M_POLICIES = {
    "short",
    "exclude",
    "replace",
    "error",
}

_VALID_BALANCED_PIXEL_STRATEGIES = {
    "random",
    "center",
    "center_closest",
}


def stable_object_seed(global_seed: int, object_id: str) -> int:
    """Derive a process-independent NumPy seed for one object."""
    digest = hashlib.sha256(
        f"{int(global_seed)}\0{str(object_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def pixel_selection_hash(indices) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validated_pixel_view(
    obj,
    *,
    object_id=None,
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
):
    """
    Return raw spectra/positions plus the ORIGINAL indices of
    analysis-valid pixels.
    """
    spectra = np.asarray(
        obj["spectra"],
        dtype=float,
    )
    positions = np.asarray(
        obj["positions_global"]
    )

    if spectra.ndim != 2:
        raise ValueError(
            f"Object {object_id!r}: spectra must be 2D, "
            f"got shape={spectra.shape}."
        )

    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or len(positions) != len(spectra)
    ):
        raise ValueError(
            f"Object {object_id!r}: positions_global must "
            "have shape (n_pixels, 2) and be aligned with spectra."
        )

    validity = spectral_pixel_validity_report(
        spectra,
        policy=pixel_validity_policy,
    )
    valid_indices = np.flatnonzero(
        validity["valid_mask"]
    ).astype(int)

    return (
        spectra,
        positions,
        valid_indices,
        validity,
    )


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
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
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
            pixel_validity_policy=pixel_validity_policy,
        )

    if level == "pixel":
        return _objects_to_pixel_matrix(
            selected,
            pixel_validity_policy=pixel_validity_policy,
        )

    if level == "balanced_pixel":
        return _objects_to_balanced_pixel_matrix(
            selected,
            m=m,
            random_state=random_state,
            replace=replace,
            balanced_pixel_strategy=balanced_pixel_strategy,
            under_m_policy=under_m_policy,
            pixel_validity_policy=pixel_validity_policy,
        )

    raise ValueError(
        "level must be one of: 'object', 'pixel', 'balanced_pixel'."
    )


def _common_object_metadata(obj_id, obj):
    return {
        "object_id": obj_id,
        "source_image": obj.get("source_clean_key", obj.get("source_image")),
        "batch": obj.get("batch"),
        "label": obj.get("object_nut_type"),
        "sample_kind": obj.get("sample_kind"),
    }


def _objects_to_object_matrix(objects, 
    spectrum_field="mean_spectrum",
    *,
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
):
    X_list = []
    rows = []
    supported_fields = {
        "mean_spectrum",
        "median_spectrum",
    }

    if spectrum_field not in supported_fields:
        raise ValueError(
            "Object-level matrices under the spectral-validity "
            "protocol support only "
            f"{sorted(supported_fields)}, got "
            f"{spectrum_field!r}."
        )

    for obj_id, obj in objects:
        spectra, _, valid_indices, _, = _validated_pixel_view(
            obj,
            object_id=str(obj_id),
            pixel_validity_policy=(
                pixel_validity_policy
            ),
        )

        if valid_indices.size == 0:
            raise ValueError(
                f"Object {obj_id!r} has no "
                "analysis-valid spectral pixel."
            )

        valid_spectra = spectra[valid_indices]

        if spectrum_field == "mean_spectrum":
            spectrum = np.mean(
                valid_spectra,
                axis=0,
            )
        elif spectrum_field == "median_spectrum":
            spectrum = np.median(
                valid_spectra,
                axis=0,
            )
        else:
            raise RuntimeError(
                "Unreachable spectrum-field branch."
            )
        X_list.append(spectrum)
        rows.append(
            _common_object_metadata(
                obj_id,
                obj,
            )
        )

    if not X_list:
        raise ValueError(
            "No objects found with the requested filters."
        )
    X = np.vstack(X_list)
    meta = _metadata_rows_to_arrays(rows)
    y = meta["label"]

    return X, y, meta


def _objects_to_pixel_matrix(objects, *, pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY):
    X_list = []
    rows = []

    for obj_id, obj in objects:
        spectra, positions, valid_indices, validity = _validated_pixel_view(
            obj, 
            object_id=str(obj_id),
            pixel_validity_policy=pixel_validity_policy,
        )
        if valid_indices.size == 0:
            raise ValueError(
                f"Object {obj_id!r} has no "
                "analysis-valid spectral pixel."
            )

        X_list.append(spectra[valid_indices])

        base = _common_object_metadata(obj_id, obj)
        for i in valid_indices:
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
    *,
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
):
    X_list = []
    rows = []

    for obj_id, obj in objects:
        spectra = np.asarray(
            obj["spectra"],
            dtype=float,
        )

        positions = np.asarray(
            obj["positions_global"]
        )

        idx = select_balanced_pixel_indices(
            obj,
            m=m,
            random_state=random_state,
            object_id=str(obj_id),
            replace=replace,
            balanced_pixel_strategy=(
                balanced_pixel_strategy
            ),
            under_m_policy=under_m_policy,
            pixel_validity_policy=(
                pixel_validity_policy
            ),
        )

        if idx is None or len(idx) == 0:
            continue

        idx = np.asarray(
            idx,
            dtype=int,
        )

        # idx already contains ORIGINAL object-pixel indices.
        X_selected = spectra[idx]
        positions_selected = positions[idx]

        X_list.append(X_selected)

        base = _common_object_metadata(
            obj_id,
            obj,
        )

        for local_i, pixel_i in enumerate(idx):
            row = base.copy()

            row["pixel_index"] = int(pixel_i)

            row["row"] = int(
                positions_selected[local_i, 0]
            )
            row["col"] = int(
                positions_selected[local_i, 1]
            )

            rows.append(row)

    if not X_list:
        raise ValueError(
            "No valid balanced pixels found "
            "with the requested filters."
        )

    X = np.vstack(X_list)
    meta = _metadata_rows_to_arrays(rows)
    y = meta["label"]

    return X, y, meta


def _metadata_rows_to_arrays(rows):
    if not rows:
        return {}

    keys = rows[0].keys()

    return {
        key: np.asarray(
            [row[key] for row in rows]
        )
        for key in keys
    }


def _resolve_under_m_policy(
    *,
    replace: bool,
    under_m_policy: str | None,
) -> str:
    """Resolve and validate the behavior for objects with fewer than m pixels."""
    policy = (
        "replace" if replace else "short"
    ) if under_m_policy is None else str(under_m_policy)

    if policy not in _VALID_UNDER_M_POLICIES:
        raise ValueError(
            "under_m_policy must be one of "
            f"{sorted(_VALID_UNDER_M_POLICIES)}, "
            f"got {policy!r}."
        )

    return policy


def _finalize_pixel_selection(
    indices,
    *,
    status: str,
    object_id,
    n_raw: int,
    n_available: int,
    seed: int | None,
    return_diagnostics: bool,
):
    """Normalize selected indices and optionally return audit diagnostics."""
    normalized = (
        None
        if indices is None
        else np.asarray(indices, dtype=int)
    )

    if not return_diagnostics:
        return normalized

    hash_values = (
        np.array([], dtype=int)
        if normalized is None
        else normalized
    )

    return normalized, {
        "object_id": (
            None
            if object_id is None
            else str(object_id)
        ),
        "n_raw": int(n_raw),
        "n_available": int(n_available),
        "n_invalid": int(n_raw - n_available),
        "n_selected": int(hash_values.size),
        "selection_hash": pixel_selection_hash(hash_values),
        "object_seed": seed,
        "status": str(status),
    }


def _select_random_pixels(
    valid_indices: np.ndarray,
    *,
    m: int,
    rng,
    under_m_policy: str,
    object_id,
) -> tuple[np.ndarray | None, str]:
    """Select random original pixel indices."""
    n_available = int(valid_indices.size)

    if n_available >= m:
        local_indices = rng.choice(
            n_available,
            size=m,
            replace=False,
        )
        return valid_indices[local_indices], "accepted"

    if under_m_policy == "replace":
        local_indices = rng.choice(
            n_available,
            size=m,
            replace=True,
        )
        return (
            valid_indices[local_indices],
            "accepted_with_replacement",
        )

    if under_m_policy == "short":
        return valid_indices.copy(), "accepted_short"

    if under_m_policy == "exclude":
        return None, "excluded_under_m"

    raise ValueError(
        f"Object {object_id!r} has {n_available} valid pixels, "
        f"fewer than requested m={m}."
    )


def _select_center_pixels(
    obj,
    positions: np.ndarray,
    valid_indices: np.ndarray,
    *,
    m: int,
    rng,
    under_m_policy: str,
    object_id,
) -> tuple[np.ndarray | None, str]:
    """Select valid pixels ordered by distance to the object centroid."""
    centroid_value = obj.get("centroid")
    centroid = (
        positions.mean(axis=0)
        if centroid_value is None
        else np.asarray(centroid_value, dtype=float)
    )

    if centroid.shape != (2,):
        raise ValueError(
            f"Object {object_id!r}: centroid must have shape (2,), "
            f"got {centroid.shape}."
        )

    valid_positions = positions[valid_indices].astype(float)
    distances = np.linalg.norm(
        valid_positions - centroid[None, :],
        axis=1,
    )
    local_order = np.argsort(
        distances,
        kind="mergesort",
    )
    ordered_indices = valid_indices[local_order]
    n_available = int(ordered_indices.size)

    if n_available >= m:
        return ordered_indices[:m], "accepted"

    if under_m_policy == "replace":
        extra = rng.choice(
            valid_indices,
            size=m - n_available,
            replace=True,
        )
        return (
            np.concatenate([ordered_indices, extra]),
            "accepted_with_replacement",
        )

    if under_m_policy == "short":
        return ordered_indices, "accepted_short"

    if under_m_policy == "exclude":
        return None, "excluded_under_m"

    raise ValueError(
        f"Object {object_id!r} has {n_available} valid pixels, "
        f"fewer than requested m={m}."
    )


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
    pixel_validity_policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
):
    """
    Select balanced pixels among analysis-valid spectra.

    Returned indices always refer to the original rows of ``obj["spectra"]``.
    """
    m = int(m)
    if m <= 0:
        raise ValueError(
            "m must be a strictly positive integer."
        )

    strategy = str(balanced_pixel_strategy)
    if strategy not in _VALID_BALANCED_PIXEL_STRATEGIES:
        raise ValueError(
            "balanced_pixel_strategy must be one of "
            f"{sorted(_VALID_BALANCED_PIXEL_STRATEGIES)}, "
            f"got {strategy!r}."
        )

    under_policy = _resolve_under_m_policy(
        replace=bool(replace),
        under_m_policy=under_m_policy,
    )

    (
        spectra,
        positions,
        valid_indices,
        _,
    ) = _validated_pixel_view(
        obj,
        object_id=object_id,
        pixel_validity_policy=pixel_validity_policy,
    )

    n_raw = int(len(spectra))
    n_available = int(valid_indices.size)

    if rng is None:
        seed = (
            int(random_state)
            if object_id is None
            else stable_object_seed(
                int(random_state),
                str(object_id),
            )
        )
        rng = np.random.default_rng(seed)
    else:
        seed = None

    if n_available == 0:
        if under_policy == "exclude":
            indices = None
            status = "excluded_no_valid_pixels"
        elif under_policy == "short":
            indices = np.array([], dtype=int)
            status = "empty_no_valid_pixels"
        else:
            raise ValueError(
                f"Object {object_id!r} contains no "
                "analysis-valid spectral pixel."
            )

        return _finalize_pixel_selection(
            indices,
            status=status,
            object_id=object_id,
            n_raw=n_raw,
            n_available=n_available,
            seed=seed,
            return_diagnostics=return_diagnostics,
        )

    if strategy == "random":
        indices, status = _select_random_pixels(
            valid_indices,
            m=m,
            rng=rng,
            under_m_policy=under_policy,
            object_id=object_id,
        )
    else:
        indices, status = _select_center_pixels(
            obj,
            positions,
            valid_indices,
            m=m,
            rng=rng,
            under_m_policy=under_policy,
            object_id=object_id,
        )

    return _finalize_pixel_selection(
        indices,
        status=status,
        object_id=object_id,
        n_raw=n_raw,
        n_available=n_available,
        seed=seed,
        return_diagnostics=return_diagnostics,
    )