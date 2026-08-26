from __future__ import annotations

import numpy as np
import pandas as pd

from src import experiment_config as expcfg

def _find_reference_wavelengths(object_db: dict, image_db: dict):
    """
    Find wavelength axis from image_db or object_db.
    """
    for img in image_db.values():
        w = img.get("wavelengths")
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.size > 0:
                return w

    for obj in object_db.values():
        w = obj.get("wavelengths")
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.size > 0:
                return w

    raise ValueError("No wavelength axis found in image_db or object_db.")


def _slice_spectral_array(value, band_mask: np.ndarray):
    """
    Slice an array if its last dimension or only dimension matches n_bands.
    Otherwise return the value unchanged.
    """
    if value is None:
        return None

    arr = np.asarray(value)
    n_bands = len(band_mask)

    if arr.ndim == 1 and arr.shape[0] == n_bands:
        return arr[band_mask].copy()

    if arr.ndim >= 2 and arr.shape[-1] == n_bands:
        return arr[..., band_mask].copy()

    return value


def select_wavelength_range_from_database(
    object_db: dict,
    image_db: dict,
    min_nm: float = 1225.0,
    max_nm: float = 1675.0,
    inclusive: bool = True,
):
    """
    Return wavelength-restricted copies of object_db and image_db.

    The object geometry is kept unchanged:
    - same masks;
    - same labels;
    - same object ids;
    - same pixel positions;
    - same train/test/projection metadata.

    Only spectral arrays are restricted:
    - image cube;
    - object spectra;
    - mean / median / std spectra;
    - cube crops if present;
    - wavelength axes.

    This is the recommended way to compare spectral windows without changing
    the object extraction protocol.
    """
    wavelengths = _find_reference_wavelengths(object_db, image_db)

    if inclusive:
        band_mask = (wavelengths >= float(min_nm)) & (wavelengths <= float(max_nm))
    else:
        band_mask = (wavelengths > float(min_nm)) & (wavelengths < float(max_nm))

    if not np.any(band_mask):
        raise ValueError(
            f"No wavelengths found in range {min_nm}–{max_nm} nm. "
            f"Available range: {wavelengths.min():.1f}–{wavelengths.max():.1f} nm."
        )

    selected_wavelengths = wavelengths[band_mask].copy()

    image_db_sel = {}

    for image_key, img in image_db.items():
        new_img = dict(img)

        if "cube" in new_img:
            new_img["cube"] = _slice_spectral_array(new_img["cube"], band_mask)

        new_img["wavelengths"] = selected_wavelengths
        new_img["n_bands_selected"] = int(len(selected_wavelengths))
        new_img["spectral_range_min_nm"] = float(selected_wavelengths.min())
        new_img["spectral_range_max_nm"] = float(selected_wavelengths.max())

        # Optional reference image for plots only.
        # We do not overwrite image_ref because the original segmentation
        # must remain unchanged for a fair comparison.
        if "cube" in new_img:
            new_img["image_ref_selected_range"] = np.nanmax(new_img["cube"], axis=2)

        image_db_sel[image_key] = new_img

    object_db_sel = {}

    for object_id, obj in object_db.items():
        new_obj = dict(obj)

        for key in [
            "spectra",
            "mean_spectrum",
            "std_spectrum",
            "median_spectrum",
            "cube_crop",
        ]:
            if key in new_obj:
                new_obj[key] = _slice_spectral_array(new_obj[key], band_mask)

        new_obj["wavelengths"] = selected_wavelengths
        new_obj["n_bands"] = int(len(selected_wavelengths))
        new_obj["n_bands_selected"] = int(len(selected_wavelengths))
        new_obj["spectral_range_min_nm"] = float(selected_wavelengths.min())
        new_obj["spectral_range_max_nm"] = float(selected_wavelengths.max())

        # Recompute spectral summaries from selected spectra.
        if "spectra" in new_obj:
            spectra = np.asarray(new_obj["spectra"], dtype=float)
            if spectra.ndim == 2 and spectra.shape[1] == len(selected_wavelengths):
                new_obj["mean_spectrum"] = np.nanmean(spectra, axis=0)
                new_obj["std_spectrum"] = np.nanstd(spectra, axis=0)
                new_obj["median_spectrum"] = np.nanmedian(spectra, axis=0)
                new_obj["n_pixels"] = int(spectra.shape[0])

        object_db_sel[object_id] = new_obj

    info = {
        "requested_min_nm": float(min_nm),
        "requested_max_nm": float(max_nm),
        "actual_min_nm": float(selected_wavelengths.min()),
        "actual_max_nm": float(selected_wavelengths.max()),
        "n_original_bands": int(len(wavelengths)),
        "n_selected_bands": int(len(selected_wavelengths)),
        "selected_band_indices": np.where(band_mask)[0].tolist(),
        "selected_wavelengths": selected_wavelengths,
    }

    return object_db_sel, image_db_sel, selected_wavelengths, info


def wavelength_selection_summary(info: dict) -> pd.DataFrame:
    """Return a one-row summary DataFrame for saving."""
    out = dict(info)
    out["selected_band_indices"] = ",".join(map(str, out["selected_band_indices"]))
    out["selected_wavelengths"] = ",".join(f"{w:.3f}" for w in out["selected_wavelengths"])
    return pd.DataFrame([out])


def spectral_pixel_validity_report(
    X,
    *,
    policy=expcfg.SPECTRAL_PIXEL_VALIDITY_POLICY,
) -> dict:
    """
    Evaluate row-wise validity of raw reflectance spectra.

    Parameters
    ----------
    X : array-like, shape (n_pixels, n_bands)
        Raw reflectance spectra.

    require_finite : bool
        Reject spectra containing NaN or Inf.

    exclude_all_zero : bool
        Reject spectra whose complete spectral vector is zero.

    all_zero_atol : float
        Absolute tolerance used to define an all-zero spectrum.

    require_strictly_positive : bool
        Reject spectra containing R <= 0.
        Appropriate when all compared preprocessing chains must remain
        compatible with absorbance.

    Returns
    -------
    report : dict
        Row-wise masks and diagnostics.
    """
    require_finite = policy.get("require_finite")
    exclude_all_zero = policy.get("exclude_all_zero")
    all_zero_atol = policy.get("all_zero_atol")
    require_strictly_positive = policy.get("require_strictly_positive")

    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D (n_pixels, n_bands), got shape={X.shape}."
        )
    if X.shape[1] == 0:
        raise ValueError(
            "X must contain at least one spectral band."
        )

    n_rows = X.shape[0]
    finite = np.isfinite(X).all(axis=1)
    all_zero = np.all(
        np.abs(X) <= float(all_zero_atol),
        axis=1,
    )
    has_nonpositive = np.any(X <= 0.0, axis=1)
    n_zero = np.count_nonzero(X == 0.0, axis=1)
    n_nonpositive = np.count_nonzero(X <= 0.0, axis=1)
    valid = np.ones(n_rows, dtype=bool)

    if require_finite:
        valid &= finite
    if exclude_all_zero:
        valid &= ~all_zero
    if require_strictly_positive:
        valid &= ~has_nonpositive

    reason = np.full(n_rows, "valid", dtype=object)

    if require_finite:
        reason[~finite] = "non_finite_spectrum"
    if exclude_all_zero:
        mask = finite & all_zero
        reason[mask] = "all_zero_spectrum"
    if require_strictly_positive:
        mask = finite & ~all_zero & has_nonpositive
        reason[mask] = "nonpositive_reflectance"

    return {
        "valid_mask": valid,
        "finite": finite,
        "all_zero": all_zero,
        "has_nonpositive": has_nonpositive,
        "n_zero": n_zero,
        "n_nonpositive": n_nonpositive,
        "reason": reason,
        "n_rows": int(X.shape[0]),
        "n_valid": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
    }