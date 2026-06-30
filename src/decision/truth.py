from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import morphology

from src.data.database import parse_image_key


def expected_position_key_for_mixture(
    mixture_clean_key: str,
    target_class: str = "peanut",
) -> str:
    """
    Convert mixture key to matching position-reference image.

    Current NIR UCO convention:
        alm3pea2 -> pea2_pos3

    This is currently peanut-specific because the position-reference images
    encode peanut positions.
    """
    if target_class != "peanut":
        raise NotImplementedError(
            "Position-reference truth is currently implemented for target_class='peanut'."
        )

    meta = parse_image_key(mixture_clean_key)

    if not meta["is_mixture"]:
        raise ValueError(f"Not a mixture key: {mixture_clean_key}")

    components = meta["components"]

    if "almond" not in components or "peanut" not in components:
        raise ValueError(
            f"Expected almond+peanut mixture, got components={list(components)}"
        )

    almond_batch = components["almond"]["batch"]
    peanut_batch = components["peanut"]["batch"]

    return f"pea{peanut_batch}_pos{almond_batch}"


def union_object_masks(
    object_db: dict,
    source_clean_key: str,
    shape,
) -> np.ndarray:
    """Build a binary mask from all objects extracted in one source image."""
    out = np.zeros(shape, dtype=bool)

    for _, obj in object_db.items():
        if obj.get("source_clean_key") != source_clean_key:
            continue

        if "mask_global" in obj:
            out |= np.asarray(obj["mask_global"], dtype=bool)

        else:
            min_row, min_col, max_row, max_col = obj["bbox"]
            out[min_row:max_row, min_col:max_col] |= np.asarray(obj["mask"], dtype=bool)

    return out


def target_truth_map_for_image(
    image_key: str,
    image_db: dict,
    object_db: dict,
    target_class: str = "peanut",
    dilation_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a pixel-level target truth map for one image.

    Returns
    -------
    truth : bool array
        True where the target class is present.

    available : bool array
        True where truth is considered available.
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    img = image_db[image_key]
    shape = img["labels"].shape
    object_area = img["labels"] > 0

    truth = np.zeros(shape, dtype=bool)
    available = object_area.copy()

    # Pure images: every object pixel has known class.
    if img.get("is_pure", False):
        truth[object_area] = img.get("nut_type") == target_class
        return truth, available

    # Position reference images: every object pixel has known class.
    if img.get("is_position_reference", False):
        truth[object_area] = img.get("nut_type") == target_class
        return truth, available

    # Mixtures: use position-reference image.
    if img.get("is_mixture", False):
        pos_key = expected_position_key_for_mixture(
            image_key,
            target_class=target_class,
        )

        if pos_key not in image_db:
            return truth, np.zeros(shape, dtype=bool)

        ref_mask = union_object_masks(
            object_db=object_db,
            source_clean_key=pos_key,
            shape=shape,
        )

        if dilation_radius and dilation_radius > 0:
            ref_mask = morphology.binary_dilation(
                ref_mask,
                footprint=morphology.disk(dilation_radius),
            )

        truth[object_area] = ref_mask[object_area]
        return truth, available

    return truth, np.zeros(shape, dtype=bool)


def peanut_truth_map_for_image(
    image_key: str,
    image_db: dict,
    object_db: dict,
    dilation_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible peanut-specific wrapper."""
    return target_truth_map_for_image(
        image_key=image_key,
        image_db=image_db,
        object_db=object_db,
        target_class="peanut",
        dilation_radius=dilation_radius,
    )


def add_pixel_truth_labels(
    pixel_df: pd.DataFrame,
    image_db: dict,
    object_db: dict,
    target_class: str = "peanut",
    dilation_radius: int = 3,
    source_col: str = "source_image",
    row_col: str = "row",
    col_col: str = "col",
    true_col: str | None = None,
    available_col: str = "truth_available",
) -> pd.DataFrame:
    """
    Add pixel-level target truth labels to a pixel dataframe.

    Default output:
        true_peanut_pixel
        truth_available
    """
    if true_col is None:
        true_col = f"true_{target_class}_pixel"

    df = pixel_df.copy()
    df[true_col] = False
    df[available_col] = False

    cache = {}

    for image_key in df[source_col].astype(str).unique():
        cache[str(image_key)] = target_truth_map_for_image(
            image_key=image_key,
            image_db=image_db,
            object_db=object_db,
            target_class=target_class,
            dilation_radius=dilation_radius,
        )

    for image_key, idx in df.groupby(source_col).groups.items():
        truth, available = cache[str(image_key)]

        rows = df.loc[idx, row_col].astype(int).to_numpy()
        cols = df.loc[idx, col_col].astype(int).to_numpy()

        df.loc[idx, true_col] = truth[rows, cols]
        df.loc[idx, available_col] = available[rows, cols]

    return df