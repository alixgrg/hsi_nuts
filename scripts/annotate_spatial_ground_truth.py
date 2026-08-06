"""Create independent peanut-presence masks for notebook 01B.

The tool deliberately loads no model, score or prediction artifact.  It can
either open a small matplotlib lasso editor or import two already prepared
NumPy layers.  Human annotations are stored as primary data under
``HSI Data/annotations/spatial_gt_v1``; notebook 01B only validates and locks
them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import experiment_config as cfg  # noqa: E402
from src.decision.truth import (  # noqa: E402
    build_spatial_ground_truth_manifest,
    validate_reference_annotation,
)
from src.protocol_governance import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate peanut presence without loading predictions."
    )
    parser.add_argument("--image", required=True, help="Source image, e.g. peanut4.")
    parser.add_argument("--annotator", required=True, help="Independent annotator id.")
    parser.add_argument(
        "--target-mask",
        type=Path,
        help="Import an existing binary peanut-presence .npy mask.",
    )
    parser.add_argument(
        "--validity-mask",
        type=Path,
        help="Import an existing binary validity .npy mask.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the canonical files without opening the editor.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing annotation for this image and annotator.",
    )
    args = parser.parse_args()
    if bool(args.target_mask) != bool(args.validity_mask):
        parser.error("--target-mask and --validity-mask must be provided together.")
    if args.validate_only and args.target_mask:
        parser.error("--validate-only cannot be combined with import paths.")
    return args


def _text_attr(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_annotation_source(image_key: str):
    database_path = PROJECT_ROOT.joinpath(*cfg.DATABASE_H5_RELATIVE_PATH)
    with h5py.File(database_path, "r") as h5:
        if image_key not in h5["images"]:
            raise KeyError(f"Unknown source image: {image_key}")
        group = h5["images"][image_key]
        image_ref = np.asarray(group["image_ref"])
        labels = np.asarray(group["labels"])
        source_class = _text_attr(group.attrs.get("nut_type", "unknown"))
    return image_ref, labels, source_class


def annotation_paths(image_key: str, annotator_id: str) -> dict[str, Path]:
    root = PROJECT_ROOT.joinpath(*cfg.SPATIAL_GT_ANNOTATION_RELATIVE_DIR)
    reference_id = f"{image_key}__{annotator_id}"
    return {
        "root": root,
        "roi": root / "roi_masks" / f"{image_key}.npy",
        "target": root / "target_masks" / f"{reference_id}.npy",
        "validity": root / "validity_masks" / f"{reference_id}.npy",
        "metadata": root / "metadata" / f"{reference_id}.json",
    }


def load_protocol() -> tuple[dict, Path, str]:
    path = PROJECT_ROOT.joinpath(*cfg.SPATIAL_GT_ANNOTATION_PROTOCOL_RELATIVE_PATH)
    protocol = json.loads(path.read_text("utf-8"))
    required = {
        "protocol_version": cfg.SPATIAL_GT_ANNOTATION_PROTOCOL_VERSION,
        "target_class": cfg.SPATIAL_GT_TARGET_CLASS,
        "annotated_class": cfg.SPATIAL_GT_ANNOTATED_CLASS,
        "positive_value": cfg.SPATIAL_GT_POSITIVE_VALUE,
        "positive_class": cfg.SPATIAL_GT_POSITIVE_CLASS,
        "positive_definition": cfg.SPATIAL_GT_POSITIVE_DEFINITION,
        "negative_value": cfg.SPATIAL_GT_NEGATIVE_VALUE,
        "negative_definition": cfg.SPATIAL_GT_NEGATIVE_DEFINITION,
        "double_annotation_policy": cfg.SPATIAL_GT_DOUBLE_ANNOTATION_POLICY,
        "annotation_tool": cfg.SPATIAL_GT_ANNOTATION_TOOL,
        "annotation_tool_version": cfg.SPATIAL_GT_ANNOTATION_TOOL_VERSION,
    }
    mismatches = {
        key: (protocol.get(key), expected)
        for key, expected in required.items()
        if protocol.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Annotation protocol/config mismatch: {mismatches}")
    nested_required = {
        "annotation_roi.source": (
            protocol.get("annotation_roi", {}).get("source"),
            cfg.SPATIAL_GT_ROI_SOURCE,
        ),
        "annotation_roi.outside_roi_definition": (
            protocol.get("annotation_roi", {}).get("outside_roi_definition"),
            cfg.SPATIAL_GT_OUTSIDE_ROI_DEFINITION,
        ),
        "boundary_policy.policy_id": (
            protocol.get("boundary_policy", {}).get("policy_id"),
            cfg.SPATIAL_GT_BOUNDARY_POLICY_ID,
        ),
        "ambiguity_policy.policy_id": (
            protocol.get("ambiguity_policy", {}).get("policy_id"),
            cfg.SPATIAL_GT_AMBIGUITY_POLICY_ID,
        ),
    }
    nested_mismatches = {
        key: (actual, expected)
        for key, (actual, expected) in nested_required.items()
        if actual != expected
    }
    if nested_mismatches:
        raise RuntimeError(
            f"Annotation protocol/config mismatch: {nested_mismatches}"
        )
    return protocol, path, sha256_file(path)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(values, dtype=bool), allow_pickle=False)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def edit_masks(image_ref: np.ndarray, roi: np.ndarray):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as PolygonPath
    from matplotlib.widgets import LassoSelector

    # Matplotlib maps ``s`` to its own "Save figure" dialog.  This editor
    # reserves the same key for committing the annotation, so remove only the
    # built-in binding while keeping the other navigation shortcuts intact.
    plt.rcParams["keymap.save"] = [
        key for key in plt.rcParams["keymap.save"] if key != "s"
    ]

    target = np.zeros(roi.shape, dtype=bool)
    validity = roi.copy()
    history: list[tuple[np.ndarray, np.ndarray]] = []
    state = {"mode": "peanut", "saved": False, "cancelled": False}
    rows, cols = np.indices(roi.shape)
    points = np.column_stack((cols.ravel(), rows.ravel()))

    finite = image_ref[np.isfinite(image_ref)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(image_ref, cmap="gray", vmin=low, vmax=high, interpolation="nearest")
    ax.contour(roi, levels=[0.5], colors="cyan", linewidths=0.5)
    overlay = ax.imshow(
        np.zeros((*roi.shape, 4), dtype=float),
        interpolation="nearest",
    )
    status = ax.set_title("")

    def refresh() -> None:
        rgba = np.zeros((*roi.shape, 4), dtype=float)
        rgba[target] = (0.0, 1.0, 0.0, 0.45)
        ambiguous = roi & ~validity
        rgba[ambiguous] = (1.0, 0.0, 1.0, 0.55)
        overlay.set_data(rgba)
        status.set_text(
            "p=peanut, n=non-peanut, a=ambiguous, u=undo, s=save, q=cancel | "
            f"mode={state['mode']} | positive={int(target.sum())} | "
            f"ambiguous={int(ambiguous.sum())}"
        )
        fig.canvas.draw_idle()

    def on_select(vertices) -> None:
        if len(vertices) < 3:
            return
        selected = PolygonPath(vertices).contains_points(points).reshape(roi.shape)
        selected &= roi
        if not np.any(selected):
            return
        history.append((target.copy(), validity.copy()))
        if state["mode"] == "peanut":
            validity[selected] = True
            target[selected] = True
        elif state["mode"] == "non_peanut":
            validity[selected] = True
            target[selected] = False
        else:
            validity[selected] = False
            target[selected] = False
        refresh()

    def on_key(event) -> None:
        if event.key == "p":
            state["mode"] = "peanut"
        elif event.key == "n":
            state["mode"] = "non_peanut"
        elif event.key == "a":
            state["mode"] = "ambiguous"
        elif event.key == "u" and history:
            previous_target, previous_validity = history.pop()
            target[:] = previous_target
            validity[:] = previous_validity
        elif event.key == "s":
            state["saved"] = True
            plt.close(fig)
            return
        elif event.key == "q":
            state["cancelled"] = True
            plt.close(fig)
            return
        refresh()

    lasso = LassoSelector(ax, on_select)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()
    lasso.disconnect_events()
    if state["cancelled"] or not state["saved"]:
        raise RuntimeError("Annotation cancelled; no file was written.")
    return target, validity


def annotation_record(
    *,
    image_key: str,
    annotator_id: str,
    source_class: str,
    labels: np.ndarray,
    paths: dict[str, Path],
    protocol_sha256: str,
    status: str,
) -> dict:
    metadata = (
        json.loads(paths["metadata"].read_text("utf-8"))
        if paths["metadata"].exists()
        else {}
    )
    return {
        "reference_id": f"{image_key}__{annotator_id}",
        "source_image": image_key,
        "source_class": source_class,
        "annotator_id": annotator_id,
        "truth_level": "pixel_annotated",
        "target_class": cfg.SPATIAL_GT_TARGET_CLASS,
        "annotated_class": cfg.SPATIAL_GT_ANNOTATED_CLASS,
        "positive_value": cfg.SPATIAL_GT_POSITIVE_VALUE,
        "positive_class": cfg.SPATIAL_GT_POSITIVE_CLASS,
        "negative_value": cfg.SPATIAL_GT_NEGATIVE_VALUE,
        "image_shape": labels.shape,
        "object_area": labels > 0,
        "roi_mask": paths["roi"],
        "target_mask": paths["target"],
        "validity_mask": paths["validity"],
        "metadata": paths["metadata"],
        "annotation_protocol_sha256": protocol_sha256,
        "annotation_date": metadata.get("annotation_date"),
        "status": status,
    }


def main() -> int:
    args = parse_args()
    image_ref, labels, source_class = load_annotation_source(args.image)
    roi = labels > 0
    protocol, protocol_path, protocol_sha256 = load_protocol()
    paths = annotation_paths(args.image, args.annotator)
    reference_id = f"{args.image}__{args.annotator}"

    if args.validate_only:
        record = annotation_record(
            image_key=args.image,
            annotator_id=args.annotator,
            source_class=source_class,
            labels=labels,
            paths=paths,
            protocol_sha256=protocol_sha256,
            status="accepted",
        )
        manifest = build_spatial_ground_truth_manifest([record])
        print(manifest.to_string(index=False))
        return 0

    protected = [paths["target"], paths["validity"], paths["metadata"]]
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError(
            f"Annotation {reference_id} already exists; use --overwrite explicitly."
        )
    if args.target_mask:
        target = np.load(args.target_mask, allow_pickle=False)
        validity = np.load(args.validity_mask, allow_pickle=False)
    else:
        target, validity = edit_masks(image_ref, roi)
    target, validity = validate_reference_annotation(
        target,
        validity,
        labels.shape,
        roi,
    )

    if paths["roi"].exists():
        stored_roi = np.load(paths["roi"], allow_pickle=False).astype(bool)
        if not np.array_equal(stored_roi, roi):
            raise RuntimeError(
                f"Frozen ROI changed for {args.image}; refuse to overwrite it."
            )
    else:
        _atomic_save_npy(paths["roi"], roi)
    _atomic_save_npy(paths["target"], target)
    _atomic_save_npy(paths["validity"], validity)
    metadata = {
        "reference_id": reference_id,
        "source_image": args.image,
        "source_class": source_class,
        "annotator_id": args.annotator,
        "target_class": cfg.SPATIAL_GT_TARGET_CLASS,
        "annotated_class": cfg.SPATIAL_GT_ANNOTATED_CLASS,
        "annotation_date": datetime.now(timezone.utc).isoformat(),
        "annotation_tool": cfg.SPATIAL_GT_ANNOTATION_TOOL,
        "annotation_tool_version": cfg.SPATIAL_GT_ANNOTATION_TOOL_VERSION,
        "annotation_protocol_path": str(protocol_path),
        "annotation_protocol_sha256": protocol_sha256,
        "mask_semantics_id": cfg.SPATIAL_GT_MASK_SEMANTICS_ID,
        "boundary_policy_id": cfg.SPATIAL_GT_BOUNDARY_POLICY_ID,
        "ambiguity_policy_id": cfg.SPATIAL_GT_AMBIGUITY_POLICY_ID,
        "roi_source": cfg.SPATIAL_GT_ROI_SOURCE,
        "roi_mask_path": str(paths["roi"]),
        "roi_sha256": sha256_file(paths["roi"]),
        "target_mask_path": str(paths["target"]),
        "target_mask_sha256": sha256_file(paths["target"]),
        "validity_mask_path": str(paths["validity"]),
        "validity_mask_sha256": sha256_file(paths["validity"]),
        "n_roi_pixels": int(roi.sum()),
        "n_valid_pixels": int(validity.sum()),
        "n_positive_pixels": int(target.sum()),
        "n_ambiguous_pixels": int((roi & ~validity).sum()),
        "protocol": protocol,
    }
    _atomic_write_json(paths["metadata"], metadata)

    record = annotation_record(
        image_key=args.image,
        annotator_id=args.annotator,
        source_class=source_class,
        labels=labels,
        paths=paths,
        protocol_sha256=protocol_sha256,
        status="accepted",
    )
    manifest = build_spatial_ground_truth_manifest([record])
    print(manifest.to_string(index=False))
    print(f"Saved independent annotation: {reference_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
