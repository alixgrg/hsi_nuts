from __future__ import annotations

import numpy as np
import pytest

from src.data.database import (
    RawManifestValidationError,
    build_raw_image_manifest,
    build_minimal_nir_uco_object_database,
    detect_known_image_keys,
    extract_objects_from_labeled_image,
    infer_object_nut_type_from_metadata,
    infer_split_from_metadata,
    parse_image_key,
    resolve_selected_keys,
    segmentation_metadata,
    validate_raw_image_manifest,
)
from src.io.database_h5 import (
    database_content_hash,
    load_nir_uco_h5,
    save_nir_uco_h5,
    validate_nir_uco_h5,
)


def test_parse_image_key_recognizes_active_nir_uco_patterns():
    pure = parse_image_key("Almond1_sb")
    assert pure["clean_key"] == "almond1"
    assert pure["sample_kind"] == "pure"
    assert pure["nut_type"] == "almond"
    assert pure["batch"] == 1
    assert pure["is_pure"] is True
    assert pure["image_status"] == "accepted"
    assert pure["metadata_status"] == "accepted"

    abbreviated_pure = parse_image_key("alm1_sb")
    assert abbreviated_pure["sample_kind"] == "pure"
    assert abbreviated_pure["nut_type"] == "almond"

    mixture = parse_image_key("alm1pea2_sb")
    assert mixture["sample_kind"] == "mixture"
    assert mixture["nut_type"] == "mixture"
    assert mixture["components"]["almond"]["batch"] == 1
    assert mixture["components"]["peanut"]["batch"] == 2

    position_reference = parse_image_key("pea3_pos2_sb")
    assert position_reference["sample_kind"] == "position_reference"
    assert position_reference["nut_type"] == "peanut"
    assert position_reference["batch"] == 3
    assert position_reference["position_set"] == 2

    unknown = parse_image_key("calibration_panel")
    assert unknown["sample_kind"] == "unknown"
    assert unknown["is_unknown"] is True
    assert unknown["image_status"] == "excluded"
    assert unknown["metadata_status"] == "error"


def test_raw_manifest_is_compact_and_unknown_cube_is_blocking():
    cube = np.ones((3, 4, 69), dtype=np.float32)
    manifest, errors = build_raw_image_manifest(
        {"alm1pea2_sb": cube, "unknown_cube": cube},
        expected_band_count=69,
        strict_scientific_role=True,
    )
    mixture = manifest.set_index("clean_key").loc["alm1pea2"]
    assert mixture["batch"] is None
    assert mixture["components_json"] == '{"almond":1,"peanut":2}'
    assert errors["original_key"].tolist() == ["unknown_cube"]
    with pytest.raises(RawManifestValidationError, match="unknown"):
        validate_raw_image_manifest(manifest)


def test_split_and_object_label_inference_are_public_and_stable():
    pure = parse_image_key("peanut2_sb")
    mixture = parse_image_key("alm1pea2_sb")
    position_reference = parse_image_key("pea2_pos1_sb")

    assert infer_split_from_metadata(pure) == "train_minimal"
    assert infer_split_from_metadata(mixture) == "projection"
    assert infer_split_from_metadata(position_reference) == "position_reference"
    assert infer_object_nut_type_from_metadata(pure) == "peanut"
    assert infer_object_nut_type_from_metadata(mixture) == "unknown"
    assert infer_object_nut_type_from_metadata(position_reference) == "peanut"


def test_extract_objects_from_labeled_image_adds_stable_object_metadata():
    cube = np.arange(4 * 4 * 5, dtype=float).reshape(4, 4, 5) + 1.0
    labels = np.zeros((4, 4), dtype=int)
    labels[1:3, 1:3] = 1
    image_ref = cube.mean(axis=2)
    wavelengths = np.linspace(900.0, 940.0, 5)
    image_meta = parse_image_key("peanut1_sb")

    objects = extract_objects_from_labeled_image(
        cube=cube,
        labels=labels,
        image_ref=image_ref,
        image_meta=image_meta,
        wavelengths=wavelengths,
        min_area=2,
    )

    assert list(objects) == ["peanut1_obj001"]
    obj = objects["peanut1_obj001"]
    assert obj["split"] == "train_minimal"
    assert obj["object_nut_type"] == "peanut"
    assert obj["spectra"].shape == (4, 5)
    np.testing.assert_allclose(obj["mean_spectrum"], obj["spectra"].mean(axis=0))
    np.testing.assert_allclose(obj["wavelengths"], wavelengths)

    seg_meta = segmentation_metadata(labels > 0, labels, threshold=0.2)
    assert seg_meta["n_labels_positive"] == 1
    assert seg_meta["mask_area_pixels"] == 4


def test_build_database_reuses_segmentation_min_area_for_object_extraction(monkeypatch):
    cube = np.ones((8, 8, 5), dtype=float)
    labels = np.zeros((8, 8), dtype=int)
    labels[0:2, 0:2] = 1
    labels[2:7, 2:7] = 2
    image_ref = cube.mean(axis=2)
    calls = []

    def fake_segment_objects(cube_arg, **kwargs):
        calls.append(kwargs)
        return image_ref, labels > 0, labels, 0.02

    monkeypatch.setattr("src.data.database.segment_objects", fake_segment_objects)

    object_db, image_db = build_minimal_nir_uco_object_database(
        data={"almond1_sb": cube},
        selected_keys=["almond1_sb"],
        segmentation_kwargs={"min_area": 5},
        skip_unknown=True,
    )

    assert calls == [{"min_area": 5}]
    assert list(object_db) == ["almond1_obj002"]
    assert object_db["almond1_obj002"]["label_id"] == 2
    assert object_db["almond1_obj002"]["area_pixels"] == 25
    assert image_db["almond1"]["n_objects"] == 1


def test_build_database_explicit_min_area_overrides_segmentation_min_area(monkeypatch):
    cube = np.ones((8, 8, 5), dtype=float)
    labels = np.zeros((8, 8), dtype=int)
    labels[0:2, 0:2] = 1
    labels[2:7, 2:7] = 2
    image_ref = cube.mean(axis=2)

    def fake_segment_objects(cube_arg, **kwargs):
        return image_ref, labels > 0, labels, 0.02

    monkeypatch.setattr("src.data.database.segment_objects", fake_segment_objects)

    object_db, _ = build_minimal_nir_uco_object_database(
        data={"almond1_sb": cube},
        selected_keys=["almond1_sb"],
        segmentation_kwargs={"min_area": 20},
        min_area=2,
        skip_unknown=True,
    )

    assert [obj["label_id"] for obj in object_db.values()] == [1, 2]


def test_detect_known_image_keys_keeps_known_hyperspectral_cubes_only():
    cube = np.ones((2, 3, 4), dtype=float)
    data = {
        "almond1_sb": cube,
        "pea1_pos1_sb": cube,
        "alm1pea2_sb": cube,
        "unknown_name": cube,
        "__header__": "not a cube",
    }

    detected = detect_known_image_keys(data)

    assert [key for key, _ in detected] == [
        "almond1_sb",
        "pea1_pos1_sb",
        "alm1pea2_sb",
    ]
    assert all(meta["is_unknown"] is False for _, meta in detected)


def test_resolve_selected_keys_accepts_raw_clean_and_suffix_variants():
    cube = np.ones((2, 3, 4), dtype=float)
    data = {
        "almond1_sb": cube,
        "peanut2_sb": cube,
        "alm1pea2_sb": cube,
    }

    assert resolve_selected_keys(data, None) is None
    assert resolve_selected_keys(data, []) is None
    assert resolve_selected_keys(data, ["almond1_sb", "peanut2", "alm1pea2"]) == [
        "almond1_sb",
        "peanut2_sb",
        "alm1pea2_sb",
    ]

    with pytest.raises(KeyError, match="not found"):
        resolve_selected_keys(data, ["walnut9"])


def test_save_load_nir_uco_h5_roundtrip_reconstructs_heavy_arrays(tmp_path, mini_hsi_db):
    object_db, image_db = mini_hsi_db
    path = tmp_path / "mini_nir_uco_database.h5"

    saved_path = save_nir_uco_h5(
        object_db,
        image_db,
        path,
        include_heavy_object_arrays=False,
    )
    assert validate_nir_uco_h5(saved_path) is True
    deep_report = validate_nir_uco_h5(
        saved_path,
        expected_image_db=image_db,
        expected_object_db=object_db,
        deep=True,
        return_report=True,
    )
    assert deep_report["passed"].all()
    assert {
        "basic_structure",
        "spatial_shapes_match",
        "spectra_shape",
        "memory_roundtrip_spectra",
    }.issubset(set(deep_report["check"]))
    loaded_object_db, loaded_image_db = load_nir_uco_h5(
        saved_path,
        reconstruct_heavy_object_arrays=True,
    )

    assert set(loaded_image_db) == set(image_db)
    assert set(loaded_object_db) == set(object_db)

    object_id = "almond1_obj001"
    loaded_obj = loaded_object_db[object_id]
    original_obj = object_db[object_id]

    np.testing.assert_allclose(loaded_obj["spectra"], original_obj["spectra"])
    np.testing.assert_allclose(loaded_obj["mean_spectrum"], original_obj["mean_spectrum"])
    np.testing.assert_array_equal(loaded_obj["mask"], original_obj["mask"])
    np.testing.assert_array_equal(loaded_obj["mask_global"], original_obj["mask_global"])
    np.testing.assert_allclose(loaded_obj["cube_crop"], image_db["almond1"]["cube"][1:3, 1:3, :])
    np.testing.assert_allclose(loaded_image_db["almond1"]["cube"], image_db["almond1"]["cube"])
    assert database_content_hash(
        loaded_object_db,
        loaded_image_db,
    ) == database_content_hash(object_db, image_db)


def test_save_nir_uco_h5_rejects_object_dtype_arrays(tmp_path, mini_hsi_db):
    object_db, image_db = mini_hsi_db
    object_db = {key: value.copy() for key, value in object_db.items()}
    object_db["almond1_obj001"]["spectra"] = np.asarray([object(), object()], dtype=object)

    with pytest.raises(TypeError, match="object-dtype"):
        save_nir_uco_h5(object_db, image_db, tmp_path / "bad.h5")
