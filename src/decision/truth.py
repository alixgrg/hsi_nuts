from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure, morphology

from src import experiment_config as expcfg
from src.data.database import parse_image_key
from src.decision.labels import (
    DEFAULT_TARGET_CLASS,
    true_col as make_true_col,
)
from src.decision.metrics import binary_mask_agreement, component_agreement
from src.protocol_governance import (
    canonical_json,
    sha256_file,
    sha256_payload,
)


@dataclass(frozen=True)
class TruthResult:
    truth_mask: np.ndarray
    available_mask: np.ndarray
    truth_level: str
    reference_id: str
    provenance: dict


def _load_reference_mask(value) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "mask" not in archive:
                    raise KeyError(f"{path} must contain an array named 'mask'.")
                return archive["mask"]
        from imageio.v3 import imread

        return imread(path)
    return np.asarray(value)


def select_annotation_subset(
    split_manifest: pd.DataFrame,
    object_summary: pd.DataFrame,
    *,
    stratify_by=("label", "batch", "size_class", "source_image"),
    fraction: float = expcfg.SPATIAL_GT_ANNOTATION_FRACTION,
    random_state: int = expcfg.RANDOM_STATE,
) -> pd.DataFrame:
    """Select a reproducible image-level annotation subset without predictions."""
    forbidden = [
        column
        for column in split_manifest.columns
        if "pred" in column.lower() or "score" in column.lower()
    ]
    if forbidden:
        raise ValueError(
            f"Model predictions/scores are forbidden during annotation selection: {forbidden}"
        )
    eligible = split_manifest[
        split_manifest["protocol_role"].eq("test")
        & split_manifest["batch"].isin(expcfg.SPATIAL_GT_TEST_BATCHES)
        & split_manifest["qc_eligibility"].eq("accepted")
    ].copy()
    if eligible.empty:
        return eligible
    summary = object_summary.copy()
    if "object_id" in summary:
        extra = [
            column
            for column in stratify_by
            if column not in eligible and column in summary
        ]
        if extra:
            eligible = eligible.merge(
                summary[["object_id", *extra]].drop_duplicates("object_id"),
                on="object_id",
                how="left",
                validate="one_to_one",
            )
    group_columns = [
        column
        for column in stratify_by
        if column in eligible.columns and column != "source_image"
    ]
    images = eligible.drop_duplicates("source_image").copy()
    rng = np.random.default_rng(int(random_state))
    selected_indices = []
    groups = (
        images.groupby(group_columns, dropna=False, sort=True)
        if group_columns
        else [(None, images)]
    )
    for _, group in groups:
        n_select = max(1, int(np.ceil(len(group) * float(fraction))))
        selected_indices.extend(
            rng.choice(group.index.to_numpy(), size=n_select, replace=False)
        )
    return (
        images.loc[np.unique(selected_indices)]
        .sort_values("source_image")
        .reset_index(drop=True)
    )


def select_double_annotation_images(
    annotation_subset: pd.DataFrame,
    *,
    policy: str = expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_POLICY,
    fraction: float = expcfg.SPATIAL_GT_DOUBLE_ANNOTATION_FRACTION,
    random_state: int = expcfg.RANDOM_STATE,
) -> set[str]:
    """Select image ids for independent double annotation.

    The active small-sample policy double-annotates every selected image.  A
    stratified fractional policy remains available for a future, larger
    annotation campaign and always keeps at least one image per source class.
    """
    if "source_image" not in annotation_subset:
        raise KeyError("annotation_subset is missing source_image.")
    images = annotation_subset.drop_duplicates("source_image").copy()
    if images.empty:
        return set()
    if policy == "all_selected_images":
        return set(images["source_image"].astype(str))
    if policy != "stratified_fraction":
        raise ValueError(f"Unknown double-annotation policy: {policy!r}")
    if not 0 < float(fraction) <= 1:
        raise ValueError("Double-annotation fraction must be in (0, 1].")
    rng = np.random.default_rng(int(random_state))
    group_columns = [column for column in ("label", "source_class") if column in images]
    groups = (
        images.groupby(group_columns, dropna=False, sort=True)
        if group_columns
        else [(None, images)]
    )
    selected = []
    for _, group in groups:
        n_select = max(1, int(np.ceil(len(group) * float(fraction))))
        selected.extend(
            rng.choice(group.index.to_numpy(), size=n_select, replace=False)
        )
    return set(images.loc[np.unique(selected), "source_image"].astype(str))


def validate_reference_mask(
    mask,
    image_shape,
    object_area=None,
) -> np.ndarray:
    """Validate an independently drawn binary reference mask."""
    values = np.asarray(mask)
    if values.shape != tuple(image_shape) or values.ndim != 2:
        raise ValueError(
            f"Reference mask shape {values.shape} != image shape {tuple(image_shape)}."
        )
    if not (
        np.issubdtype(values.dtype, np.bool_)
        or set(np.unique(values)).issubset({0, 1})
    ):
        raise ValueError("Reference mask must be binary.")
    result = values.astype(bool, copy=False)
    if object_area is not None:
        roi = np.asarray(object_area, dtype=bool)
        if roi.shape != result.shape:
            raise ValueError("Object-area ROI does not match reference mask.")
        outside = result & ~roi
        if np.any(outside):
            raise ValueError(
                f"Reference mask contains {int(outside.sum())} pixels outside the object ROI."
            )
    return result


def validate_reference_annotation(
    target_mask,
    validity_mask,
    image_shape,
    object_area,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate peanut-presence and validity layers on the annotation ROI."""
    roi = np.asarray(object_area, dtype=bool)
    target = validate_reference_mask(target_mask, image_shape, roi)
    validity = validate_reference_mask(validity_mask, image_shape, roi)
    invalid_positive = target & ~validity
    if np.any(invalid_positive):
        raise ValueError(
            "Target mask contains positive peanut pixels without a valid "
            f"binary decision: {int(invalid_positive.sum())} pixel(s)."
        )
    return target, validity


def extract_reference_components(
    mask,
    *,
    reference_id: str | None = None,
    connectivity: int = expcfg.SPATIAL_GT_COMPONENT_CONNECTIVITY,
) -> pd.DataFrame:
    """Return one compact row per connected reference fragment."""
    labels = measure.label(np.asarray(mask, dtype=bool), connectivity=connectivity)
    rows = []
    for region in measure.regionprops(labels):
        rows.append(
            {
                "reference_id": reference_id,
                "component_id": int(region.label),
                "area_pixels": int(region.area),
                "centroid_row": float(region.centroid[0]),
                "centroid_col": float(region.centroid[1]),
                "bbox_json": canonical_json(tuple(int(x) for x in region.bbox)),
            }
        )
    return pd.DataFrame(rows, columns=expcfg.SPATIAL_GT_COMPONENT_COLUMNS)


def _annotation_semantics(record: dict) -> dict:
    values = {
        "target_class": str(
            record.get("target_class", expcfg.SPATIAL_GT_TARGET_CLASS)
        ),
        "annotated_class": str(
            record.get("annotated_class", expcfg.SPATIAL_GT_ANNOTATED_CLASS)
        ),
        "positive_value": int(
            record.get("positive_value", expcfg.SPATIAL_GT_POSITIVE_VALUE)
        ),
        "positive_class": str(
            record.get("positive_class", expcfg.SPATIAL_GT_POSITIVE_CLASS)
        ),
        "positive_definition": str(
            record.get(
                "positive_definition",
                expcfg.SPATIAL_GT_POSITIVE_DEFINITION,
            )
        ),
        "negative_value": int(
            record.get("negative_value", expcfg.SPATIAL_GT_NEGATIVE_VALUE)
        ),
        "negative_definition": str(
            record.get(
                "negative_definition",
                expcfg.SPATIAL_GT_NEGATIVE_DEFINITION,
            )
        ),
        "outside_roi_definition": str(
            record.get(
                "outside_roi_definition",
                expcfg.SPATIAL_GT_OUTSIDE_ROI_DEFINITION,
            )
        ),
        "mask_semantics_id": str(
            record.get(
                "mask_semantics_id",
                expcfg.SPATIAL_GT_MASK_SEMANTICS_ID,
            )
        ),
        "boundary_policy_id": str(
            record.get(
                "boundary_policy_id",
                expcfg.SPATIAL_GT_BOUNDARY_POLICY_ID,
            )
        ),
        "ambiguity_policy_id": str(
            record.get(
                "ambiguity_policy_id",
                expcfg.SPATIAL_GT_AMBIGUITY_POLICY_ID,
            )
        ),
    }
    expected = {
        "target_class": expcfg.SPATIAL_GT_TARGET_CLASS,
        "annotated_class": expcfg.SPATIAL_GT_ANNOTATED_CLASS,
        "positive_value": expcfg.SPATIAL_GT_POSITIVE_VALUE,
        "positive_class": expcfg.SPATIAL_GT_POSITIVE_CLASS,
        "positive_definition": expcfg.SPATIAL_GT_POSITIVE_DEFINITION,
        "negative_value": expcfg.SPATIAL_GT_NEGATIVE_VALUE,
        "negative_definition": expcfg.SPATIAL_GT_NEGATIVE_DEFINITION,
        "outside_roi_definition": expcfg.SPATIAL_GT_OUTSIDE_ROI_DEFINITION,
        "mask_semantics_id": expcfg.SPATIAL_GT_MASK_SEMANTICS_ID,
        "boundary_policy_id": expcfg.SPATIAL_GT_BOUNDARY_POLICY_ID,
        "ambiguity_policy_id": expcfg.SPATIAL_GT_AMBIGUITY_POLICY_ID,
    }
    mismatches = {
        key: (values[key], expected_value)
        for key, expected_value in expected.items()
        if values[key] != expected_value
    }
    if mismatches:
        raise ValueError(f"Annotation semantics do not match the protocol: {mismatches}")
    return values


def build_spatial_ground_truth_manifest(annotation_records) -> pd.DataFrame:
    """Validate peanut-presence annotations and produce their provenance manifest."""
    rows = []
    for record in annotation_records:
        truth_level = str(record.get("truth_level", "pixel_annotated"))
        if truth_level not in expcfg.SPATIAL_GT_ALLOWED_LEVELS:
            raise ValueError(f"Unknown truth level: {truth_level!r}")
        semantics = _annotation_semantics(record)
        reference_id = str(record["reference_id"])
        source_image = str(record["source_image"])
        source_class = str(record["source_class"])
        annotator_id = str(record["annotator_id"])
        roi = np.asarray(record["object_area"], dtype=bool)
        if roi.shape != tuple(record["image_shape"]) or roi.ndim != 2:
            raise ValueError("Annotation ROI does not match the image shape.")
        protocol_sha256 = str(record.get("annotation_protocol_sha256", "")).strip()
        if not protocol_sha256:
            raise ValueError("annotation_protocol_sha256 is required.")
        paths = {
            "roi_mask_path": Path(record["roi_mask"]),
            "target_mask_path": Path(record["target_mask"]),
            "validity_mask_path": Path(record["validity_mask"]),
            "metadata_path": Path(record["metadata"]),
        }
        requested_status = str(record.get("status", "pending"))
        all_files_exist = all(path.exists() for path in paths.values())
        if requested_status == "accepted" and not all_files_exist:
            missing = [str(path) for path in paths.values() if not path.exists()]
            raise FileNotFoundError(f"Accepted annotation is missing files: {missing}")
        status = "accepted" if requested_status == "accepted" else "pending"
        base = {
            "reference_id": reference_id,
            "source_image": source_image,
            "source_class": source_class,
            "annotator_id": annotator_id,
            "truth_level": truth_level,
            **semantics,
            "annotation_tool": str(
                record.get("annotation_tool", expcfg.SPATIAL_GT_ANNOTATION_TOOL)
            ),
            "annotation_tool_version": str(
                record.get(
                    "annotation_tool_version",
                    expcfg.SPATIAL_GT_ANNOTATION_TOOL_VERSION,
                )
            ),
            "annotation_protocol_version": str(
                record.get(
                    "annotation_protocol_version",
                    expcfg.SPATIAL_GT_ANNOTATION_PROTOCOL_VERSION,
                )
            ),
            "annotation_protocol_sha256": protocol_sha256,
            "annotation_date": record.get("annotation_date"),
            "roi_source": str(
                record.get("roi_source", expcfg.SPATIAL_GT_ROI_SOURCE)
            ),
            "roi_mask_path": str(paths["roi_mask_path"]),
            "roi_sha256": None,
            "target_mask_path": str(paths["target_mask_path"]),
            "target_mask_sha256": None,
            "validity_mask_path": str(paths["validity_mask_path"]),
            "validity_mask_sha256": None,
            "metadata_path": str(paths["metadata_path"]),
            "metadata_sha256": None,
            "n_roi_pixels": int(roi.sum()),
            "n_valid_pixels": np.nan,
            "n_positive_pixels": np.nan,
            "n_ambiguous_pixels": np.nan,
            "status": status,
        }
        if status == "pending":
            rows.append(base)
            continue
        stored_roi = validate_reference_mask(
            _load_reference_mask(paths["roi_mask_path"]),
            record["image_shape"],
        )
        if not np.array_equal(stored_roi, roi):
            raise ValueError("Stored annotation ROI differs from image_db labels > 0.")
        target, validity = validate_reference_annotation(
            _load_reference_mask(paths["target_mask_path"]),
            _load_reference_mask(paths["validity_mask_path"]),
            record["image_shape"],
            roi,
        )
        metadata = json.loads(paths["metadata_path"].read_text("utf-8"))
        metadata_expected = {
            "reference_id": reference_id,
            "source_image": source_image,
            "source_class": source_class,
            "annotator_id": annotator_id,
            "target_class": expcfg.SPATIAL_GT_TARGET_CLASS,
            "annotation_protocol_sha256": protocol_sha256,
        }
        metadata_mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in metadata_expected.items()
            if str(metadata.get(key)) != str(expected)
        }
        if metadata_mismatches:
            raise ValueError(
                f"Annotation metadata do not match the manifest: {metadata_mismatches}"
            )
        base.update(
            {
                "annotation_date": str(metadata.get("annotation_date", "")),
                "roi_sha256": sha256_file(paths["roi_mask_path"]),
                "target_mask_sha256": sha256_file(paths["target_mask_path"]),
                "validity_mask_sha256": sha256_file(paths["validity_mask_path"]),
                "metadata_sha256": sha256_file(paths["metadata_path"]),
                "n_valid_pixels": int(validity.sum()),
                "n_positive_pixels": int(target.sum()),
                "n_ambiguous_pixels": int((roi & ~validity).sum()),
            }
        )
        rows.append(base)
    result = pd.DataFrame(rows, columns=expcfg.SPATIAL_GT_MANIFEST_COLUMNS)
    if not result.empty and result["reference_id"].duplicated().any():
        raise ValueError("reference_id values must be unique.")
    return result


def resolve_truth_for_image(
    image_key: str,
    image_db: dict,
    object_db: dict,
    *,
    annotation_manifest: pd.DataFrame | None = None,
    reference_masks: dict | None = None,
    reference_validity_masks: dict | None = None,
    target_class: str = expcfg.SPATIAL_GT_TARGET_CLASS,
    dilation_radius: int = 3,
) -> TruthResult:
    """Resolve the strongest available truth and state its evidence level."""
    if annotation_manifest is not None and not annotation_manifest.empty:
        candidates = annotation_manifest[
            annotation_manifest["source_image"].astype(str).eq(str(image_key))
            & annotation_manifest["truth_level"].eq("pixel_annotated")
            & annotation_manifest["status"].eq("accepted")
            & annotation_manifest["target_class"].astype(str).eq(str(target_class))
        ]
        if not candidates.empty:
            row = candidates.sort_values("reference_id").iloc[0]
            reference_id = str(row["reference_id"])
            source = (
                reference_masks[reference_id]
                if reference_masks and reference_id in reference_masks
                else row["target_mask_path"]
            )
            validity_source = (
                reference_validity_masks[reference_id]
                if reference_validity_masks
                and reference_id in reference_validity_masks
                else row["validity_mask_path"]
            )
            mask, validity = validate_reference_annotation(
                _load_reference_mask(source),
                _load_reference_mask(validity_source),
                image_db[image_key]["labels"].shape,
                image_db[image_key]["labels"] > 0,
            )
            return TruthResult(
                truth_mask=mask,
                available_mask=validity,
                truth_level="pixel_annotated",
                reference_id=reference_id,
                provenance=row.to_dict(),
            )
    truth, available = target_truth_map_for_image(
        image_key,
        image_db,
        object_db,
        target_class=target_class,
        dilation_radius=dilation_radius,
    )
    image = image_db[image_key]
    if image.get("is_pure") or image.get("is_position_reference"):
        level = "weak_object_label"
        reference_id = f"object-label:{image_key}"
    elif image.get("is_mixture"):
        level = "indirect"
        reference_id = f"position-proxy:{image_key}"
    else:
        level = "indirect"
        reference_id = f"unavailable:{image_key}"
    return TruthResult(
        truth_mask=truth,
        available_mask=available,
        truth_level=level,
        reference_id=reference_id,
        provenance={"method": "legacy_position_or_object_label"},
    )


def build_annotation_agreement_table(
    annotation_manifest: pd.DataFrame,
    *,
    reference_masks: dict | None = None,
    reference_validity_masks: dict | None = None,
    reference_rois: dict | None = None,
) -> pd.DataFrame:
    """Compare double peanut annotations on jointly valid ROI pixels."""
    rows = []
    for source_image, group in annotation_manifest.groupby("source_image"):
        accepted = group[
            group["truth_level"].eq("pixel_annotated")
            & group["status"].eq("accepted")
        ]
        if len(accepted) < 2:
            continue
        pair = accepted.sort_values("reference_id").iloc[:2]
        target_masks = []
        validity_masks = []
        roi_source = (
            reference_rois.get(str(source_image))
            if reference_rois and str(source_image) in reference_rois
            else pair.iloc[0]["roi_mask_path"]
        )
        roi = np.asarray(_load_reference_mask(roi_source), dtype=bool)
        for _, record in pair.iterrows():
            reference_id = str(record["reference_id"])
            target_source = (
                reference_masks[reference_id]
                if reference_masks and reference_id in reference_masks
                else record["target_mask_path"]
            )
            validity_source = (
                reference_validity_masks[reference_id]
                if reference_validity_masks
                and reference_id in reference_validity_masks
                else record["validity_mask_path"]
            )
            target, validity = validate_reference_annotation(
                _load_reference_mask(target_source),
                _load_reference_mask(validity_source),
                roi.shape,
                roi,
            )
            target_masks.append(target)
            validity_masks.append(validity)
        pairwise_valid = roi & validity_masks[0] & validity_masks[1]
        n_roi_pixels = int(roi.sum())
        n_pairwise_valid = int(pairwise_valid.sum())
        if n_roi_pixels == 0:
            raise ValueError(f"Annotation ROI is empty for {source_image}.")
        if n_pairwise_valid == 0:
            raise RuntimeError(
                f"No jointly valid annotated pixels for {source_image}."
            )
        pixel = binary_mask_agreement(
            target_masks[0],
            target_masks[1],
            roi=pairwise_valid,
        )
        pixel.pop("n_roi_pixels", None)
        components = component_agreement(
            target_masks[0] & pairwise_valid,
            target_masks[1] & pairwise_valid,
            connectivity=expcfg.SPATIAL_GT_COMPONENT_CONNECTIVITY,
        )
        status = (
            "accepted"
            if pixel["dice"] >= expcfg.SPATIAL_AGREEMENT_MIN_DICE
            and pixel["iou"] >= expcfg.SPATIAL_AGREEMENT_MIN_IOU
            and components["unmatched_component_rate"]
            <= expcfg.SPATIAL_AGREEMENT_MAX_UNMATCHED_COMPONENT_RATE
            else "adjudication_required"
        )
        rows.append(
            {
                "source_image": str(source_image),
                "target_class": str(pair.iloc[0]["target_class"]),
                "reference_id_a": str(pair.iloc[0]["reference_id"]),
                "reference_id_b": str(pair.iloc[1]["reference_id"]),
                "n_roi_pixels": n_roi_pixels,
                "n_pairwise_valid_pixels": n_pairwise_valid,
                "pairwise_valid_coverage": float(
                    n_pairwise_valid / n_roi_pixels
                ),
                "ambiguous_rate_a": float(
                    np.count_nonzero(roi & ~validity_masks[0]) / n_roi_pixels
                ),
                "ambiguous_rate_b": float(
                    np.count_nonzero(roi & ~validity_masks[1]) / n_roi_pixels
                ),
                "validity_agreement": float(
                    np.mean(validity_masks[0][roi] == validity_masks[1][roi])
                ),
                **pixel,
                **components,
                "dice_passed": (
                    pixel["dice"] >= expcfg.SPATIAL_AGREEMENT_MIN_DICE
                ),
                "iou_passed": (
                    pixel["iou"] >= expcfg.SPATIAL_AGREEMENT_MIN_IOU
                ),
                "unmatched_rate_passed": (
                    components["unmatched_component_rate"]
                    <= expcfg.SPATIAL_AGREEMENT_MAX_UNMATCHED_COMPONENT_RATE
                ),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=expcfg.SPATIAL_GT_AGREEMENT_COLUMNS)


def validate_annotation_adjudication(
    agreement: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> None:
    required = {
        "source_image",
        "status",
        "adjudicator",
        "date",
        "justification",
        "reference_id",
    }
    missing = required.difference(adjudication.columns)
    if missing:
        raise ValueError(f"Adjudication is missing columns: {sorted(missing)}")
    needed = set(
        agreement.loc[
            agreement["status"].eq("adjudication_required"),
            "source_image",
        ].astype(str)
    )
    decisions = adjudication[
        adjudication["source_image"].astype(str).isin(needed)
    ]
    if set(decisions["source_image"].astype(str)) != needed:
        raise RuntimeError("Every disagreement must be adjudicated.")
    valid = (
        decisions["status"].eq("adjudicated")
        & decisions[
            ["adjudicator", "date", "justification", "reference_id"]
        ].notna().all(axis=1)
        & decisions[
            ["adjudicator", "date", "justification", "reference_id"]
        ].astype(str).apply(lambda column: column.str.strip().ne("")).all(axis=1)
    )
    if not bool(valid.all()):
        raise RuntimeError("Adjudication contains incomplete or pending rows.")


def build_spatial_ground_truth_lock(
    annotation_manifest: pd.DataFrame,
    component_manifest: pd.DataFrame,
    agreement: pd.DataFrame,
    adjudication: pd.DataFrame,
    *,
    configuration_hash: str,
) -> dict:
    """Freeze all independent annotations and their compact manifests."""
    if annotation_manifest.empty:
        raise RuntimeError("Cannot lock an empty annotation manifest.")
    if annotation_manifest["status"].isin({"pending"}).any():
        raise RuntimeError("Pending annotations cannot be locked.")
    validate_annotation_adjudication(agreement, adjudication)
    annotation_file_hashes = {
        str(row["reference_id"]): {
            "roi_mask": str(row["roi_sha256"]),
            "target_mask": str(row["target_mask_sha256"]),
            "validity_mask": str(row["validity_mask_sha256"]),
            "metadata": str(row["metadata_sha256"]),
        }
        for _, row in annotation_manifest.iterrows()
    }
    payload = {
        "protocol_version": expcfg.PROTOCOL_VERSION,
        "annotation_protocol_version": (
            expcfg.SPATIAL_GT_ANNOTATION_PROTOCOL_VERSION
        ),
        "configuration_hash": str(configuration_hash),
        "annotation_file_hashes": annotation_file_hashes,
        "annotation_manifest_sha256": sha256_payload(
            annotation_manifest.to_dict("records")
        ),
        "component_manifest_sha256": sha256_payload(
            component_manifest.to_dict("records")
        ),
        "agreement_sha256": sha256_payload(agreement.to_dict("records")),
        "adjudication_sha256": sha256_payload(
            adjudication.to_dict("records")
        ),
    }
    payload["lock_sha256"] = sha256_payload(payload)
    return payload


def verify_spatial_ground_truth_lock(
    lock,
    annotation_manifest: pd.DataFrame,
    component_manifest: pd.DataFrame,
    agreement: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> None:
    """Block if any mask or table changed after the spatial lock."""
    payload = json.loads(Path(lock).read_text("utf-8")) if isinstance(
        lock, (str, Path)
    ) else dict(lock)
    expected = dict(payload)
    lock_hash = expected.pop("lock_sha256", None)
    if lock_hash != sha256_payload(expected):
        raise RuntimeError("Spatial ground-truth lock JSON was modified.")
    actual_table_hashes = {
        "annotation_manifest_sha256": sha256_payload(
            annotation_manifest.to_dict("records")
        ),
        "component_manifest_sha256": sha256_payload(
            component_manifest.to_dict("records")
        ),
        "agreement_sha256": sha256_payload(agreement.to_dict("records")),
        "adjudication_sha256": sha256_payload(
            adjudication.to_dict("records")
        ),
    }
    for key, actual in actual_table_hashes.items():
        if str(payload.get(key)) != actual:
            raise RuntimeError(f"Spatial artifact changed after lock: {key}")
    file_columns = {
        "roi_mask": "roi_mask_path",
        "target_mask": "target_mask_path",
        "validity_mask": "validity_mask_path",
        "metadata": "metadata_path",
    }
    for _, row in annotation_manifest.iterrows():
        reference_id = str(row["reference_id"])
        expected_hashes = payload["annotation_file_hashes"].get(
            reference_id, {}
        )
        for file_kind, path_column in file_columns.items():
            path = row.get(path_column)
            actual = sha256_file(path)
            if expected_hashes.get(file_kind) != actual:
                raise RuntimeError(
                    "Mask changed after lock: "
                    f"reference_id={reference_id}, file={file_kind}"
                )

def expected_position_key_for_mixture(
    mixture_clean_key: str,
    target_class: str = "peanut",
) -> str:
    """
    Convert mixture key to matching position-reference image.

    Current NIR UCO convention:
        alm3pea2 -> pea2_pos3

    This resolver is specific to peanut position-reference images.
    """
    if target_class != "peanut":
        raise NotImplementedError(
            "NIR UCO position-reference truth is currently implemented "
            "only for target_class='peanut'."
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
    target_class: str = DEFAULT_TARGET_CLASS,
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


def pure_image_class_truth(
    image_key: str,
    image_db: dict,
    *,
    target_class: str = expcfg.TARGET_CLASS,
    allowed_batches=(1, 2),
) -> TruthResult:
    """Return exact in-mask pixel truth for a pure reference image.

    A pure image supplies class truth automatically: every segmented nut pixel
    has the image class and background is unavailable. The function blocks on
    mixtures, unknown classes, forbidden batches, or missing segmentation.
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")
    image = image_db[image_key]
    # Accept either canonical boolean metadata or the explicit sample kind.
    is_pure = bool(image.get("is_pure", False)) or str(
        image.get("sample_kind", "")
    ).strip().lower() == "pure"
    if not is_pure:
        raise RuntimeError(f"Spatial calibration image is not pure: {image_key}")
    batch = int(image.get("batch"))
    if batch not in set(map(int, allowed_batches)):
        raise RuntimeError(
            f"Spatial calibration image {image_key} belongs to batch {batch}."
        )
    class_name = str(image.get("nut_type", image.get("object_nut_type", "")))
    if class_name not in {"almond", "peanut"}:
        raise RuntimeError(
            f"Unknown pure-image class for {image_key}: {class_name!r}"
        )
    if "labels" not in image:
        raise RuntimeError(f"Missing segmentation labels for {image_key}.")
    available = np.asarray(image["labels"]) > 0
    if not available.any():
        raise RuntimeError(f"Empty segmented ROI for {image_key}.")
    truth = available & (class_name == str(target_class))
    return TruthResult(
        truth_mask=truth,
        available_mask=available,
        truth_level="pure_image_class_exact",
        reference_id=f"pure-image:{image_key}",
        provenance={
            "method": "pure_image_class_within_segmented_roi",
            "source_image": str(image_key),
            "batch": batch,
            "class_name": class_name,
        },
    )


def peanut_truth_map_for_image(
    image_key: str,
    image_db: dict,
    object_db: dict,
    dilation_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
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
    target_class: str = DEFAULT_TARGET_CLASS,
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
        true_target_class_pixel
        truth_available
    """
    if true_col is None:
        true_col = make_true_col(target_class, "pixel")

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
