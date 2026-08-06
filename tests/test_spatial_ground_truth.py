from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import experiment_config as expcfg
from src.decision.metrics import binary_mask_agreement, component_agreement
from src.decision.truth import (
    build_annotation_agreement_table,
    build_spatial_ground_truth_lock,
    build_spatial_ground_truth_manifest,
    extract_reference_components,
    resolve_truth_for_image,
    select_double_annotation_images,
    validate_reference_annotation,
    validate_reference_mask,
    verify_spatial_ground_truth_lock,
)


def _annotation_record(
    tmp_path,
    reference_id,
    annotator_id,
    target,
    validity,
    roi,
):
    roi_path = tmp_path / "image__roi.npy"
    target_path = tmp_path / f"{reference_id}__target.npy"
    validity_path = tmp_path / f"{reference_id}__validity.npy"
    metadata_path = tmp_path / f"{reference_id}.json"
    np.save(roi_path, roi)
    np.save(target_path, target)
    np.save(validity_path, validity)
    metadata_path.write_text(
        json.dumps(
            {
                "reference_id": reference_id,
                "source_image": "image",
                "source_class": "peanut",
                "annotator_id": annotator_id,
                "target_class": "peanut",
                "annotation_protocol_sha256": "protocol",
                "annotation_date": "2026-08-03T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return {
        "reference_id": reference_id,
        "source_image": "image",
        "source_class": "peanut",
        "annotator_id": annotator_id,
        "truth_level": "pixel_annotated",
        "target_class": "peanut",
        "annotated_class": "peanut",
        "positive_value": 1,
        "positive_class": "peanut",
        "negative_value": 0,
        "roi_mask": roi_path,
        "target_mask": target_path,
        "validity_mask": validity_path,
        "metadata": metadata_path,
        "annotation_protocol_sha256": "protocol",
        "image_shape": roi.shape,
        "object_area": roi,
        "status": "accepted",
    }


def test_reference_mask_contract_and_truth_level(mini_hsi_db):
    object_db, image_db = mini_hsi_db
    shape = image_db["almond1"]["labels"].shape
    mask = image_db["almond1"]["labels"] > 0
    np.testing.assert_array_equal(
        validate_reference_mask(mask, shape, mask),
        mask,
    )
    with pytest.raises(ValueError, match="shape"):
        validate_reference_mask(mask[:-1], shape)
    target = mask.copy()
    validity = mask.copy()
    row, col = np.argwhere(mask)[0]
    validity[row, col] = False
    target[row, col] = True
    with pytest.raises(ValueError, match="without a valid binary decision"):
        validate_reference_annotation(target, validity, shape, mask)
    result = resolve_truth_for_image("almond1", image_db, object_db)
    assert result.truth_level == "weak_object_label"
    assert result.reference_id == "object-label:almond1"


def test_every_selected_image_is_double_annotated():
    subset = pd.DataFrame(
        {
            "source_image": ["almond4", "peanut4"],
            "label": ["almond", "peanut"],
        }
    )
    assert select_double_annotation_images(subset) == {"almond4", "peanut4"}
    assert expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION == 1.0


def test_mask_and_component_agreement_are_separate_constraints():
    mask_a = np.zeros((8, 8), dtype=bool)
    mask_b = np.zeros((8, 8), dtype=bool)
    mask_a[1:3, 1:3] = True
    mask_a[5:7, 5:7] = True
    mask_b[1:3, 1:3] = True
    mask_b[5:7, 4:6] = True

    pixel = binary_mask_agreement(mask_a, mask_b)
    components = component_agreement(mask_a, mask_b)

    assert pixel["pixel_agreement"] == pytest.approx(60 / 64)
    assert pixel["dice"] == pytest.approx(0.75)
    assert pixel["iou"] == pytest.approx(0.6)
    assert components["n_components_matched"] == 2
    assert components["mean_matched_component_iou"] == pytest.approx(2 / 3)
    assert components["unmatched_component_rate"] == 0.0


def test_pending_annotations_and_post_lock_modification_are_blocking(tmp_path):
    shape = (8, 8)
    roi = np.ones(shape, dtype=bool)
    target = np.zeros(shape, dtype=bool)
    target[2:6, 2:6] = True
    validity = roi.copy()

    pending = build_spatial_ground_truth_manifest(
        [
            {
                "reference_id": "missing",
                "source_image": "image",
                "source_class": "peanut",
                "annotator_id": "a",
                "truth_level": "pixel_annotated",
                "roi_mask": tmp_path / "missing_roi.npy",
                "target_mask": tmp_path / "missing_target.npy",
                "validity_mask": tmp_path / "missing_validity.npy",
                "metadata": tmp_path / "missing.json",
                "annotation_protocol_sha256": "protocol",
                "image_shape": shape,
                "object_area": roi,
                "status": "pending",
            }
        ]
    )
    assert pending.loc[0, "status"] == "pending"
    assert pending.loc[0, "target_class"] == "peanut"
    assert pending.loc[0, "positive_definition"]
    assert pending.loc[0, "outside_roi_definition"]

    records = [
        _annotation_record(
            tmp_path,
            reference_id,
            annotator,
            target,
            validity,
            roi,
        )
        for reference_id, annotator in (
            ("image__a", "a"),
            ("image__b", "b"),
        )
    ]
    manifest = build_spatial_ground_truth_manifest(records)
    agreement = build_annotation_agreement_table(manifest)
    assert agreement.loc[0, "pairwise_valid_coverage"] == 1.0
    assert agreement.loc[0, "target_class"] == "peanut"
    components = pd.concat(
        [
            extract_reference_components(target, reference_id="image__a"),
            extract_reference_components(target, reference_id="image__b"),
        ],
        ignore_index=True,
    )
    adjudication = pd.DataFrame(columns=expcfg.SPATIAL_GT_ADJUDICATION_COLUMNS)
    lock = build_spatial_ground_truth_lock(
        manifest,
        components,
        agreement,
        adjudication,
        configuration_hash="configuration",
    )
    verify_spatial_ground_truth_lock(
        lock,
        manifest,
        components,
        agreement,
        adjudication,
    )

    changed = target.copy()
    changed[0, 0] = True
    np.save(records[0]["target_mask"], changed)
    with pytest.raises(RuntimeError, match="Mask changed"):
        verify_spatial_ground_truth_lock(
            lock,
            manifest,
            components,
            agreement,
            adjudication,
        )
