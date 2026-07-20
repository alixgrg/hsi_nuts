from __future__ import annotations

import numpy as np
import pytest


def _mini_object(
    *,
    object_id: str,
    source_clean_key: str,
    label: str,
    base_value: float,
):
    spectra = np.asarray(
        [
            [base_value + 0.01, base_value + 0.02, base_value + 0.03, base_value + 0.04],
            [base_value + 0.02, base_value + 0.03, base_value + 0.04, base_value + 0.05],
            [base_value + 0.03, base_value + 0.04, base_value + 0.05, base_value + 0.06],
            [base_value + 0.04, base_value + 0.05, base_value + 0.06, base_value + 0.07],
        ],
        dtype=float,
    )
    positions = np.asarray([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=int)
    mask_crop = np.ones((2, 2), dtype=bool)
    mask_global = np.zeros((4, 4), dtype=bool)
    mask_global[1:3, 1:3] = True

    return {
        "object_id": object_id,
        "object_index": 1,
        "label_id": 1,
        "source_image": f"{source_clean_key}_sb",
        "source_clean_key": source_clean_key,
        "sample_kind": "pure",
        "image_nut_type": label,
        "object_nut_type": label,
        "batch": 1,
        "components": {label: {"batch": 1, "token": label[:3]}},
        "position_set": None,
        "is_pure": True,
        "is_mixture": False,
        "is_position_reference": False,
        "is_unknown": False,
        "split": "train_minimal",
        "bbox": (1, 1, 3, 3),
        "centroid": (1.5, 1.5),
        "area_pixels": 4,
        "mask": mask_crop,
        "mask_global": mask_global,
        "positions_global": positions,
        "positions_local": positions - np.asarray([1, 1]),
        "cube_crop": np.ones((2, 2, 4), dtype=float) * base_value,
        "image_ref_crop": np.ones((2, 2), dtype=float) * base_value,
        "spectra": spectra,
        "mean_spectrum": spectra.mean(axis=0),
        "std_spectrum": spectra.std(axis=0),
        "median_spectrum": np.median(spectra, axis=0),
        "wavelengths": np.asarray([900.0, 910.0, 920.0, 930.0]),
        "data_mode": "reflectance",
        "n_pixels": 4,
        "n_bands": 4,
        "description": f"pure {label}, batch 1",
    }


@pytest.fixture
def mini_hsi_db():
    almond = _mini_object(
        object_id="almond1_obj001",
        source_clean_key="almond1",
        label="almond",
        base_value=0.20,
    )
    peanut = _mini_object(
        object_id="peanut1_obj001",
        source_clean_key="peanut1",
        label="peanut",
        base_value=0.50,
    )

    object_db = {
        almond["object_id"]: almond,
        peanut["object_id"]: peanut,
    }
    image_db = {}
    for source_clean_key, obj in [("almond1", almond), ("peanut1", peanut)]:
        cube = np.ones((4, 4, 4), dtype=float) * float(obj["spectra"].mean())
        labels = np.zeros((4, 4), dtype=int)
        labels[1:3, 1:3] = 1
        mask = labels > 0
        image_db[source_clean_key] = {
            "image_id": f"{source_clean_key}_sb",
            "clean_key": source_clean_key,
            "original_key": f"{source_clean_key}_sb",
            "sample_kind": "pure",
            "nut_type": obj["object_nut_type"],
            "batch": 1,
            "components": obj["components"],
            "position_set": None,
            "is_pure": True,
            "is_mixture": False,
            "is_position_reference": False,
            "is_unknown": False,
            "description": obj["description"],
            "cube": cube,
            "image_ref": cube.mean(axis=2),
            "mask": mask,
            "labels": labels,
            "threshold": 0.1,
            "wavelengths": obj["wavelengths"],
            "data_mode": "reflectance",
            "n_objects": 1,
            "object_ids": [obj["object_id"]],
        }
    return object_db, image_db
