import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation, feature

from src.protocol_governance import sha256_ndarray


def make_reference_image(cube, method="median", band_index=None):
    """
    Build a 2D reference image from a hyperspectral cube.

    cube shape:
        X ∈ R^{H x W x B}

    Available methods:
    - mean   : I(i,j) = mean_b X(i,j,b)
    - median : I(i,j) = median_b X(i,j,b)
    - max    : I(i,j) = max_b X(i,j,b)
    - norm   : I(i,j) = ||X(i,j,:)||_2
    - band   : I(i,j) = X(i,j,band_index)
    """
    if method == "mean":
        return np.nanmean(cube, axis=2)

    if method == "median":
        return np.nanmedian(cube, axis=2)

    if method == "max":
        return np.nanmax(cube, axis=2)

    if method == "norm":
        return np.linalg.norm(cube, axis=2)

    if method == "band":
        if band_index is None:
            raise ValueError("band_index must be provided when method='band'.")
        return cube[:, :, band_index]

    raise ValueError("Unknown reference method.")


def make_binary_mask(
    image_ref,
    threshold_method="fixed",
    percentile=10,
    tau=None,
    tau_min=0.02,
):
    """
    Create binary mask M(i,j).

    Methods:
    - fixed:
        M(i,j) = 1 if I(i,j) > tau

    - otsu:
        tau chosen automatically by Otsu

    - otsu_min:
        tau = max(tau_otsu, tau_min)

    - percentile:
        tau = percentile of image intensities
    """
    img = np.asarray(image_ref)
    valid_pixels = img[np.isfinite(img)]

    if threshold_method == "fixed":
        if tau is None:
            tau = tau_min
        mask = img > tau

    elif threshold_method == "otsu":
        tau = filters.threshold_otsu(valid_pixels)
        mask = img > tau

    elif threshold_method == "otsu_min":
        tau_otsu = filters.threshold_otsu(valid_pixels)
        tau = max(tau_otsu, tau_min)
        mask = img > tau

    elif threshold_method == "percentile":
        tau = np.nanpercentile(img, percentile)
        mask = img > tau

    else:
        raise ValueError(
            "threshold_method must be 'fixed', 'otsu', 'otsu_min' or 'percentile'."
        )

    return mask, tau


def clean_mask(
    mask,
    min_area=20,
    opening_radius=0,
    closing_radius=1,
    fill_holes=True,
):
    """
    Clean binary mask.

    For current images, avoid being too aggressive:
    - small min_area
    - opening_radius=0 or 1
    - closing_radius=1 or 2
    """
    mask = mask.astype(bool)

    if opening_radius > 0:
        mask = morphology.opening(mask, morphology.disk(opening_radius))

    if closing_radius > 0:
        mask = morphology.closing(mask, morphology.disk(closing_radius))

    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    mask = morphology.remove_small_objects(mask, min_size=min_area)

    return mask


def label_objects_with_watershed(mask, min_distance=10):
    """
    Split touching objects with watershed.
    """
    distance = ndi.distance_transform_edt(mask)

    coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=mask,
    )

    markers = np.zeros(distance.shape, dtype=int)

    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx

    labels = segmentation.watershed(
        -distance,
        markers,
        mask=mask,
    )

    return labels


def segment_objects(
    cube,
    reference_method="max",
    threshold_method="fixed",
    percentile=10,
    tau=None,
    tau_min=0.02,
    band_index=None,
    min_area=10,
    opening_radius=0,
    closing_radius=1,
    fill_holes=True,
    use_watershed=False,
    min_distance=10,
    override_labels=None,
    override_provenance=None,
    return_provenance=False,
):
    """
    Segment individual objects in a hyperspectral cube.

    Recommended starting values:
        reference_method="max"
        threshold_method="fixed"
        tau_min=0.02
        min_area=10
        opening_radius=0
        closing_radius=1
        use_watershed=False
    """
    image_ref = make_reference_image(
        cube,
        method=reference_method,
        band_index=band_index,
    )

    if override_labels is not None:
        labels = np.asarray(override_labels)
        if labels.ndim != 2 or labels.shape != np.asarray(cube).shape[:2]:
            raise ValueError(
                "override_labels must be a 2-D label image matching the cube."
            )
        if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
            raise ValueError(
                "override_labels must contain non-negative integer labels."
            )
        if not override_provenance:
            raise ValueError(
                "override_labels requires documented override_provenance."
            )
        labels = labels.astype(np.int32, copy=False)
        mask = labels > 0
        tau = None
        provenance = dict(override_provenance)
        provenance.setdefault("source", "documented_override")
        provenance.setdefault("hash", sha256_ndarray(labels))
    else:
        mask, tau = make_binary_mask(
            image_ref,
            threshold_method=threshold_method,
            percentile=percentile,
            tau=tau,
            tau_min=tau_min,
        )

        mask = clean_mask(
            mask,
            min_area=min_area,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            fill_holes=fill_holes,
        )

        if use_watershed:
            labels = label_objects_with_watershed(
                mask,
                min_distance=min_distance,
            )
        else:
            labels = measure.label(mask)
        provenance = {
            "source": "automatic",
            "hash": sha256_ndarray(labels),
        }

    if return_provenance:
        return {
            "image_ref": image_ref,
            "mask": mask,
            "labels": labels,
            "threshold": tau,
            "provenance": provenance,
        }
    return image_ref, mask, labels, tau
