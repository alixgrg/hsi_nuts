from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from skimage import measure

from src import experiment_config as expcfg
from src.data.segmentation import segment_objects
from src.protocol_governance import (
    canonical_json,
    sha256_file,
    sha256_ndarray,
    sha256_payload,
)


class RawManifestValidationError(RuntimeError):
    """Raised when the canonical raw-data inventory is not admissible."""


class SegmentationValidationError(RuntimeError):
    """Raised when segmentation or object extraction violates its contract."""

NIR_UCO_NAME_CONFIG = {
    "suffixes_to_ignore": ["_sb"],
    # Add nuts here if needed in the future.
    "nut_aliases": {
        "almond": "almond",
        "alm": "almond",
        "peanut": "peanut",
        "pea": "peanut",
        "walnut": "walnut",
        "wal": "walnut",
    },
    # Kept for compatibility; parsing is driven by nut_aliases below.
    "patterns": {
        "pure": r"^(?P<nut_token>[a-zA-Z]+)(?P<batch>\d+)$",
        "mixture": r"^(?P<components>(?:[a-zA-Z]+\d+){2,})$",
        "position_reference": r"^(?P<nut_token>[a-zA-Z]+)(?P<batch>\d+)_pos(?P<position_set>\d+)$",
    },
}



def _remove_known_suffixes(name, suffixes):
    clean = name
    for suffix in suffixes:
        suffix = str(suffix).lower()
        if suffix and clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    return clean
    
def _empty_metadata(original_key, clean_key):
    """
    Standardised metadata for unrecognized image names
    """
    return {
        "original_key": original_key,
        "clean_key": clean_key,
        # general image type
        "sample_kind": "unknown",
        # classification type for pure samples
        "nut_type": "unknown",
        "batch": None,
        # general structure for n types of nuts and mixtures
        "components": {},
        "position_set": None,
        # useful flags
        "is_pure": False,
        "is_mixture": False,
        "is_position_reference": False,
        "is_unknown": True,
        # Initial protocol/QC status. Recognised images are promoted to
        # ``accepted`` by ``parse_image_key``; unknown names remain excluded.
        "image_status": "excluded",
        "metadata_status": "error",
        "metadata_warning": "",
        "metadata_error": "unknown image name pattern",
        # readable description
        "description": "unknown image name pattern",
    }


def _normalise_nut_token(token, nut_aliases):
    token = str(token).strip().lower()
    return nut_aliases.get(token)


def _token_batch_pairs(component_string, nut_aliases):
    component_string = str(component_string).strip().lower()
    pairs = re.findall(r"([a-zA-Z]+)(\d+)", component_string)
    if not pairs:
        return []

    reconstructed = "".join(f"{token}{batch}" for token, batch in pairs).lower()
    if reconstructed != component_string:
        return []

    out = []
    for token, batch in pairs:
        nut_type = _normalise_nut_token(token, nut_aliases)
        if nut_type is None:
            return []
        out.append(
            {
                "nut_type": nut_type,
                "batch": int(batch),
                "token": token.lower(),
            }
        )
    return out


def _parse_components(component_string, nut_aliases, min_components=2):
    """
    Parse mixture component string for instance:
        alm1pea2
        alm1pea2wal3
        pea1wal2

    return dict with structure :
        {
            "almond": {"batch": 1, "token": "alm"},
            "peanut": {"batch": 2, "token": "pea"},
            "walnut": {"batch": 3, "token": "wal"},
        }
    """
    pairs = _token_batch_pairs(component_string, nut_aliases)
    if len(pairs) < int(min_components):
        raise ValueError(f"Expected at least {min_components} components in: {component_string}")

    components = {}
    for item in pairs:
        nut_type = item["nut_type"]
        if nut_type in components:
            raise ValueError(f"Duplicate nut component in image key: {nut_type}")
        components[nut_type] = {
            "batch": item["batch"],
            "token": item["token"],
        }
    return components


def infer_split_from_metadata(meta):
    """
    Define default split role of the image in the pipeline
    """
    if meta["is_pure"]:
        return "train_minimal"
    if meta["is_mixture"]:
        return "projection"
    if meta["is_position_reference"]:
        return "position_reference"
    return "unknown"

def infer_object_nut_type_from_metadata(meta):
    """
    Deduce the nut type label of the objects in the image when it is known

    - pure image almond1 -> every object is almond
    - pure image peanut1 -> every object is peanut
    - position reference image pea1_pos2 -> objects are peanut
    - mixture image alm1pea2 -> object label unknown at extraction time
    """
    if meta["is_pure"]:
        return meta["nut_type"]
    if meta["is_position_reference"]:
        return meta["nut_type"]
    if meta["is_mixture"]:
        return "unknown"
    return "unknown"



def _unpack_segmentation_result(seg_result):
    """
    Allow handling of multiple styles of return from segment_objects.

    Expected cases :
        image_ref, mask, labels, tau
    Or dictionary :
        {
            "image_ref": ...,
            "mask": ...,
            "labels": ...,
            "threshold": ...
        }
    """
    if isinstance(seg_result, dict):
        missing = [key for key in ("image_ref", "mask", "labels") if key not in seg_result]
        if missing:
            raise ValueError(f"segmentation result is missing required keys: {missing}")
        return (
            seg_result["image_ref"],
            seg_result["mask"],
            seg_result["labels"],
            seg_result.get("threshold", None),
            dict(seg_result.get("provenance", {})),
        )
    if isinstance(seg_result, tuple) and len(seg_result) == 4:
        image_ref, mask, labels, tau = seg_result
        return image_ref, mask, labels, tau, {}
    raise ValueError(
        "segment_objects must return either "
        "(image_ref, mask, labels, threshold) "
        "or a dict with keys image_ref, mask, labels."
    )


def _validate_cube_and_segmentation(cube, labels, image_ref, mask=None):
    cube = np.asarray(cube)
    labels = np.asarray(labels)
    image_ref = np.asarray(image_ref)

    if cube.ndim != 3:
        raise ValueError(f"cube must be a 3D array, got shape={cube.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape={labels.shape}")
    if image_ref.ndim != 2:
        raise ValueError(f"image_ref must be a 2D array, got shape={image_ref.shape}")
    if labels.shape != cube.shape[:2]:
        raise ValueError(
            f"labels shape {labels.shape} does not match cube spatial shape {cube.shape[:2]}"
        )
    if image_ref.shape != cube.shape[:2]:
        raise ValueError(
            f"image_ref shape {image_ref.shape} does not match cube spatial shape {cube.shape[:2]}"
        )
    if mask is not None and np.asarray(mask).shape != cube.shape[:2]:
        raise ValueError(
            f"mask shape {np.asarray(mask).shape} does not match cube spatial shape {cube.shape[:2]}"
        )
    return cube, labels, image_ref


def segmentation_metadata(mask, labels, threshold=None):
    """Return stable image-level segmentation metadata."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels)
    return {
        "threshold": threshold,
        "n_labels_positive": int(len(np.unique(labels[labels > 0]))),
        "max_label": int(labels.max()) if labels.size else 0,
        "mask_area_pixels": int(mask.sum()),
        "mask_area_ratio": float(mask.sum() / mask.size) if mask.size else np.nan,
        "segmentation_shape": tuple(int(v) for v in labels.shape),
    }


def preprocess_nir_uco_cube(
    raw_cube,
    n_remove_start:int,
    n_stop_end:int | None = None,
):
    """
    Minimal preprocessing for NIR UCO cubes

    Remove first noisy bands:
        X_clean = X_raw[:, :, 6:]
    Extend later.
    """
    raw_cube = np.asarray(raw_cube)
    if raw_cube.ndim != 3:
        raise ValueError(f"raw_cube must be a 3D array, got shape={raw_cube.shape}")
    if int(n_remove_start) < 0:
        raise ValueError("n_remove_start must be non-negative")
    cube = raw_cube[:, :, n_remove_start:n_stop_end] if n_stop_end is not None else raw_cube[:, :, n_remove_start:]
    if cube.shape[2] == 0:
        raise ValueError("band trimming produced a cube with zero spectral bands")
    return cube


def parse_image_key(key, config=None):
    """
    Parse image name and return standardised metadata.

    Supported patterns are driven by ``nut_aliases`` in the configuration:
    - pure images: almond1, alm1, peanut2
    - mixtures: alm1pea2, almond1peanut2
    - position references: pea1_pos3, peanut1_pos3
    - suffixes to ignore: _sb
    """
    if config is None:
        config = NIR_UCO_NAME_CONFIG
    config = deepcopy(config)
    original_key = str(key)
    name = original_key.strip().lower()
    clean_key = _remove_known_suffixes(
        name,
        suffixes=config.get("suffixes_to_ignore", []),
    )
    meta = _empty_metadata(
        original_key=original_key,
        clean_key=clean_key,
    )
    nut_aliases = config["nut_aliases"]
    patterns = config.get("patterns", NIR_UCO_NAME_CONFIG["patterns"])

    # Position references must be parsed before pure samples because both
    # start with a single token/batch pair.
    pos_match = re.fullmatch(patterns["position_reference"], clean_key)
    if pos_match:
        nut_token = pos_match.group("nut_token")
        nut_type = _normalise_nut_token(nut_token, nut_aliases)
        if nut_type is not None:
            batch = int(pos_match.group("batch"))
            position_set = int(pos_match.group("position_set"))
            meta.update({
                "sample_kind": "position_reference",
                "nut_type": nut_type,
                "batch": batch,
                "components": {
                    nut_type: {
                        "batch": batch,
                        "token": nut_token.lower(),
                    }
                },
                "position_set": position_set,
                "is_position_reference": True,
                "is_unknown": False,
                "image_status": "accepted",
                "metadata_status": "accepted",
                "metadata_warning": "",
                "metadata_error": "",
                "description": (
                    f"{nut_type} batch {batch} in position set {position_set}"
                ),
            })
            return meta

    # Pure images
    pure_match = re.fullmatch(patterns["pure"], clean_key)
    if pure_match:
        nut_token = pure_match.group("nut_token")
        nut_type = _normalise_nut_token(nut_token, nut_aliases)
        if nut_type is not None:
            batch = int(pure_match.group("batch"))
            meta.update({
                "sample_kind": "pure",
                "nut_type": nut_type,
                "batch": batch,
                "components": {
                    nut_type: {
                        "batch": batch,
                        "token": nut_token.lower(),
                    }
                },
                "is_pure": True,
                "is_unknown": False,
                "image_status": "accepted",
                "metadata_status": "accepted",
                "metadata_warning": "",
                "metadata_error": "",
                "description": f"pure {nut_type}, batch {batch}",
            })
            return meta
    
    # Mixtures
    mixture_match = re.fullmatch(patterns["mixture"], clean_key)
    if mixture_match:
        try:
            components = _parse_components(clean_key, nut_aliases, min_components=2)
        except ValueError:
            return meta
        component_desc = " + ".join(
            f"{nut} batch {info['batch']}"
            for nut, info in components.items()
        )
        meta.update({
            "sample_kind": "mixture",
            "nut_type": "mixture",
            "batch": None,
            "components": components,
            "is_mixture": True,
            "is_unknown": False,
            "image_status": "accepted",
            "metadata_status": "accepted",
            "metadata_warning": "",
            "metadata_error": "",
            "description": f"mixture: {component_desc}",
        })
        return meta

    return meta


def _scientific_role_from_metadata(meta) -> str:
    if meta.get("is_mixture"):
        return "mixture_application"
    if meta.get("is_position_reference"):
        return "spatial_reference"
    if meta.get("is_pure"):
        batch = meta.get("batch")
        if batch in expcfg.PROTOCOL_CALIBRATION_BATCHES:
            return "calibration"
        if batch in expcfg.PROTOCOL_VALIDATION_BATCHES:
            return "validation"
        if batch in expcfg.PROTOCOL_TEST_BATCHES:
            return "test"
    return "unknown"


def _compact_components(components) -> dict[str, int]:
    return {
        str(nut): int(info["batch"])
        for nut, info in sorted(dict(components or {}).items())
        if isinstance(info, dict) and info.get("batch") is not None
    }


def build_raw_image_manifest(
    raw_data,
    *,
    expected_band_count: int | None = None,
    strict_scientific_role: bool = True,
):
    """Inventory every 3-D HSI cube without expanding spectra into columns."""
    rows = []
    errors = []
    for original_key, value in raw_data.items():
        cube = np.asarray(value)
        if cube.ndim != 3:
            continue
        meta = parse_image_key(original_key)
        role = _scientific_role_from_metadata(meta)
        metadata_status = str(meta.get("metadata_status", "error"))
        if expected_band_count is not None and cube.shape[2] != int(
            expected_band_count
        ):
            metadata_status = "error"
            metadata_error = (
                f"expected {int(expected_band_count)} bands, "
                f"observed {int(cube.shape[2])}"
            )
        else:
            metadata_error = str(meta.get("metadata_error", ""))
        if strict_scientific_role and role == "unknown":
            metadata_status = "error"
            metadata_error = metadata_error or "unknown scientific role"

        numeric = np.issubdtype(cube.dtype, np.number)
        if not numeric:
            metadata_status = "error"
            metadata_error = metadata_error or "cube dtype is not numeric"
        n_nan = int(np.isnan(cube).sum()) if numeric else 0
        n_inf = int(np.isinf(cube).sum()) if numeric else 0
        row = {
            "original_key": str(original_key),
            "clean_key": str(meta["clean_key"]),
            "sample_kind": str(meta["sample_kind"]),
            "scientific_role": role,
            "nut_type": str(meta["nut_type"]),
            "batch": meta.get("batch"),
            "components_json": canonical_json(
                _compact_components(meta.get("components"))
            ),
            "height": int(cube.shape[0]),
            "width": int(cube.shape[1]),
            "n_bands": int(cube.shape[2]),
            "dtype": str(cube.dtype),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "metadata_status": metadata_status,
        }
        rows.append(row)
        if metadata_status != "accepted":
            errors.append(
                {
                    "original_key": str(original_key),
                    "clean_key": str(meta["clean_key"]),
                    "metadata_status": metadata_status,
                    "metadata_error": metadata_error,
                }
            )

    manifest = pd.DataFrame(
        rows,
        columns=expcfg.RAW_IMAGE_MANIFEST_COLUMNS,
    ).sort_values("clean_key", ignore_index=True)
    parsing_errors = pd.DataFrame(
        errors,
        columns=expcfg.METADATA_PARSING_ERROR_COLUMNS,
    )
    return manifest, parsing_errors


def validate_raw_image_manifest(
    raw_manifest: pd.DataFrame,
    *,
    require_finite: bool = True,
    require_known_role: bool = True,
    require_common_band_count: bool = True,
) -> bool:
    """Fail closed when the raw HSI inventory violates the protocol."""
    missing = [
        column
        for column in expcfg.RAW_IMAGE_MANIFEST_COLUMNS
        if column not in raw_manifest.columns
    ]
    failures = []
    if missing:
        failures.append(f"missing_columns={missing}")
    if raw_manifest.empty:
        failures.append("no_hyperspectral_cube")
    if not missing:
        if raw_manifest["original_key"].duplicated().any():
            failures.append("duplicate_original_key")
        if raw_manifest["clean_key"].duplicated().any():
            failures.append("duplicate_clean_key")
        if require_finite and (
            pd.to_numeric(raw_manifest["n_nan"], errors="coerce").fillna(1).gt(0)
            | pd.to_numeric(raw_manifest["n_inf"], errors="coerce").fillna(1).gt(0)
        ).any():
            failures.append("non_finite_cube")
        if require_known_role and (
            raw_manifest["scientific_role"].eq("unknown")
            | ~raw_manifest["metadata_status"].eq("accepted")
        ).any():
            failures.append("unknown_scientific_role")
        if (
            require_common_band_count
            and raw_manifest["n_bands"].nunique(dropna=False) != 1
        ):
            failures.append("inconsistent_band_count")
    if failures:
        raise RawManifestValidationError(
            "Invalid raw image manifest: " + ", ".join(failures)
        )
    return True


def load_segmentation_override(
    override_directory,
    clean_key: str,
    *,
    expected_shape=None,
):
    """Load one documented label override and return labels plus provenance."""
    if override_directory is None:
        return None, None
    path = Path(override_directory) / f"{clean_key}.npz"
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=False) as payload:
        missing = {
            "labels",
            "justification",
            "version",
        }.difference(payload.files)
        if missing:
            raise SegmentationValidationError(
                f"Undocumented segmentation override {path}: "
                f"missing={sorted(missing)}"
            )
        labels = np.asarray(payload["labels"])
        justification = str(np.asarray(payload["justification"]).item()).strip()
        version = str(np.asarray(payload["version"]).item()).strip()
    if not justification or not version:
        raise SegmentationValidationError(
            f"Override {path} requires non-empty justification and version."
        )
    if labels.ndim != 2:
        raise SegmentationValidationError(
            f"Override {path} labels must be 2-D, got {labels.shape}."
        )
    if expected_shape is not None and tuple(labels.shape) != tuple(expected_shape):
        raise SegmentationValidationError(
            f"Override {path} shape {labels.shape} != {tuple(expected_shape)}."
        )
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise SegmentationValidationError(
            f"Override {path} must contain non-negative integer labels."
        )
    return labels.astype(np.int32, copy=False), {
        "source": "documented_override",
        "hash": sha256_file(path),
        "justification": justification,
        "version": version,
        "path": str(path),
    }


def validate_extracted_object(obj, image_record) -> bool:
    """Validate one extracted object immediately against its source image."""
    spectra = np.asarray(obj.get("spectra"))
    positions = np.asarray(obj.get("positions_global"))
    mask = np.asarray(obj.get("mask"), dtype=bool)
    labels = np.asarray(image_record.get("labels"))
    label_id = int(obj.get("label_id", -1))
    expected_mask = labels == label_id
    expected_pixels = int(expected_mask.sum())
    failures = []
    if spectra.ndim != 2:
        failures.append(f"spectra_shape={spectra.shape}")
    if positions.shape != (expected_pixels, 2):
        failures.append(f"positions_shape={positions.shape}")
    if spectra.ndim == 2 and spectra.shape[0] != expected_pixels:
        failures.append(f"spectra_rows={spectra.shape[0]}")
    if int(obj.get("n_pixels", -1)) != expected_pixels:
        failures.append(f"n_pixels={obj.get('n_pixels')}")
    if int(obj.get("area_pixels", -1)) != expected_pixels:
        failures.append(f"area_pixels={obj.get('area_pixels')}")
    if int(mask.sum()) != expected_pixels:
        failures.append(f"mask_pixels={int(mask.sum())}")
    if spectra.ndim == 2 and spectra.shape[1] != np.asarray(
        image_record.get("cube")
    ).shape[2]:
        failures.append("spectral_band_mismatch")
    if spectra.size and not np.isfinite(spectra).all():
        failures.append("non_finite_spectra")
    if failures:
        raise SegmentationValidationError(
            f"Invalid extracted object {obj.get('object_id')}: {failures}"
        )
    return True



def extract_objects_from_labeled_image(
    cube,
    labels,
    image_ref,
    image_meta,
    wavelengths=None,
    data_mode="reflectance",
    min_area=100,
    split=None,
):
    """
    Extract individual nut objects from a labelled image.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube with shape (H, W, B), already preprocessed.
    labels : np.ndarray
        Label image with shape (H, W).
        0 = background, 1..K = detected objects.
    image_ref : np.ndarray
        2D reference image used for segmentation.
    image_meta : dict
        Metadata returned by parse_image_key.
    wavelengths : np.ndarray or None
        Wavelength axis after band removal.
    data_mode : str
        "reflectance" or "absorbance".
    min_area : int
        Minimum object area in pixels.
    split : str or None
        Optional split label. If None, inferred from image metadata.

    Returns
    -------
    objects : dict
        Dictionary of extracted objects.
    """
    cube, labels, image_ref = _validate_cube_and_segmentation(cube, labels, image_ref)
    if split is None:
        split = infer_split_from_metadata(image_meta)
    object_nut_type = infer_object_nut_type_from_metadata(image_meta)
    objects = {}
    regions = measure.regionprops(labels, intensity_image=image_ref)

    for region in regions:
        if region.area < min_area:
            continue
        label_id = region.label
        min_row, min_col, max_row, max_col = region.bbox
        object_mask_global = labels == label_id
        object_mask_crop = object_mask_global[
            min_row:max_row,
            min_col:max_col,
        ]
        cube_crop = cube[
            min_row:max_row,
            min_col:max_col,
            :
        ]
        image_ref_crop = image_ref[
            min_row:max_row,
            min_col:max_col,
        ]
        # OMEGA_k = pixels positions of the object k
        positions_global = np.argwhere(object_mask_global)
        positions_local = positions_global - np.array([min_row, min_col])
        # Object spectral matrix, shape: n_object_pixels x n_bands.
        spectra = cube[object_mask_global]
        # spectral statistics
        mean_spectrum = np.nanmean(spectra, axis=0)
        std_spectrum = np.nanstd(spectra, axis=0)
        median_spectrum = np.nanmedian(spectra, axis=0)
        object_id = f"{image_meta['clean_key']}_obj{int(label_id):03d}"
        if object_id in objects:
            raise SegmentationValidationError(
                f"Duplicate deterministic object_id: {object_id}"
            )
        objects[object_id] = {
            # object identification
            "object_id": object_id,
            "object_index": int(label_id),
            "label_id": int(label_id),
            # Image source
            "source_image": image_meta["original_key"],
            "source_clean_key": image_meta["clean_key"],
            # General metadata inherited from the image
            "sample_kind": image_meta["sample_kind"],
            "image_nut_type": image_meta["nut_type"],
            "batch": image_meta["batch"],
            "components": image_meta["components"],
            "position_set": image_meta["position_set"],
            # Inherited flags
            "is_pure": image_meta["is_pure"],
            "is_mixture": image_meta["is_mixture"],
            "is_position_reference": image_meta["is_position_reference"],
            "is_unknown": image_meta["is_unknown"],
            "image_status": image_meta.get("image_status", "accepted"),
            "object_status": (
                "excluded"
                if image_meta.get("image_status") == "excluded"
                else "accepted"
            ),
            # object label for learning / projection
            # For mixtures : unknown before SIMCA prediction
            "object_nut_type": object_nut_type,
            # Split pipeline
            "split": split,
            # Geometry
            "bbox": (
                int(min_row),
                int(min_col),
                int(max_row),
                int(max_col),
            ),
            "centroid": tuple(float(v) for v in region.centroid),
            "area_pixels": int(region.area),
            # Masks and positions
            "mask": object_mask_crop,
            "mask_global": object_mask_global,
            "positions_global": positions_global,
            "positions_local": positions_local,
            # Image data
            "cube_crop": cube_crop,
            "image_ref_crop": image_ref_crop,
            # Spectral data
            "spectra": spectra,
            "mean_spectrum": mean_spectrum,
            "std_spectrum": std_spectrum,
            "median_spectrum": median_spectrum,
            # Spectral metadata
            "wavelengths": wavelengths,
            "data_mode": data_mode,
            "n_pixels": spectra.shape[0],
            "n_bands": spectra.shape[1],
            # Practical description
            "description": image_meta["description"],
        }
    return objects


def build_minimal_nir_uco_object_database(
    data,
    selected_keys=None,
    config=None,
    preprocess_func=None,
    n_remove_start=6,
    n_stop_end=None,
    wavelengths=None,
    data_mode="reflectance",
    min_area=None,
    split=None,
    skip_unknown=False,
    segmentation_kwargs=None,
    segmentation_overrides_dir=None,
):
    """
    Build a minimal object-level database from NIR UCO images.

    Parameters
    ----------
    data : dict
        Dictionary loaded from NIR_uco_sb.mat.
    selected_keys : list[str] or None
        Image names to process.
        Example: ["almond1", "almond2", "peanut1", "peanut2"].
    config : dict or None
        Parsing config used by parse_image_key.
    preprocess_func : callable or None
        Function applied to each raw cube before segmentation.
        Example: preprocess_nir_uco_cube.
        If None, cubes are used as they are.
    n_remove_start : int
        Number of initial bands to remove if using preprocess_nir_uco_cube.
    n_stop_end : int or None
        Index of the last band to keep if using preprocess_nir_uco_cube.
    wavelengths : np.ndarray or None
        Wavelength axis after preprocessing.
    data_mode : str
        "reflectance" or "absorbance".
    min_area : int or None
        Minimum area for extracted objects. If None, the value is resolved
        from ``segmentation_kwargs["min_area"]`` when available, otherwise
        the legacy default of 100 pixels is used.
    split : str or None
        Optional split label for all objects. If None, inferred from image metadata.
    skip_unknown : bool
        If True, skip image names not recognized by parse_image_key.
    segmentation_kwargs : dict or None
        Parameters passed to segment_objects.
    segmentation_overrides_dir : str or Path or None
        Directory containing documented ``<clean_key>.npz`` label overrides.

    Returns
    -------
    object_database : dict
        Object-level database.
    image_database : dict
        Image-level database with cube, mask, labels, metadata.
    """
    if selected_keys is None:
        selected_keys = list(data.keys())
    if segmentation_kwargs is None:
        segmentation_kwargs = {}
    else:
        segmentation_kwargs = dict(segmentation_kwargs)
    object_min_area = (
        min_area
        if min_area is not None
        else segmentation_kwargs.get("min_area", 100)
    )
    object_min_area = int(object_min_area)
    if object_min_area < 0:
        raise ValueError("min_area must be non-negative")

    object_database = {}
    image_database = {}

    for key in selected_keys:
        if key not in data:
            print(f"[WARNING] Key not found in data: {key}")
            continue

        image_meta = parse_image_key(key, config=config)
        if image_meta["is_unknown"] and skip_unknown:
            print(f"[WARNING] Skipping unknown image name: {key}")
            continue
        if image_meta["is_unknown"]:
            raise RawManifestValidationError(
                f"Unknown HSI cube is blocking in canonical mode: {key}"
            )
        print(
            f"Processing {key} | "
            f"kind={image_meta['sample_kind']} | "
            f"components={image_meta['components']}"
        )

        raw_cube = data[key]
        if not is_hyperspectral_cube(raw_cube):
            print(f"[WARNING] Skipping non-hyperspectral entry: {key}")
            continue

        if preprocess_func is not None:
            cube = preprocess_func(raw_cube, n_remove_start=n_remove_start, n_stop_end=n_stop_end)
        else:
            cube = np.asarray(raw_cube)
        override_labels, override_provenance = load_segmentation_override(
            segmentation_overrides_dir,
            image_meta["clean_key"],
            expected_shape=cube.shape[:2],
        )
        if override_labels is None:
            seg_result = segment_objects(cube, **segmentation_kwargs)
        else:
            seg_result = segment_objects(
                cube,
                override_labels=override_labels,
                override_provenance=override_provenance,
                return_provenance=True,
                **segmentation_kwargs,
            )

        image_ref, mask, labels, tau, provenance = _unpack_segmentation_result(
            seg_result
        )
        if not provenance:
            provenance = {
                "source": "automatic",
                "hash": sha256_ndarray(labels),
            }
        _validate_cube_and_segmentation(cube, labels, image_ref, mask=mask)
        if int(np.asarray(labels).max(initial=0)) == 0:
            raise SegmentationValidationError(
                f"Scientific image {image_meta['clean_key']} has an empty mask."
            )
        seg_meta = segmentation_metadata(mask=mask, labels=labels, threshold=tau)
        objects = extract_objects_from_labeled_image(
            cube=cube,
            labels=labels,
            image_ref=image_ref,
            image_meta=image_meta,
            wavelengths=wavelengths,
            data_mode=data_mode,
            min_area=object_min_area,
            split=split,
        )
        if not objects:
            raise SegmentationValidationError(
                f"Scientific image {image_meta['clean_key']} produced no "
                "eligible extracted object."
            )
        duplicate_ids = set(object_database).intersection(objects)
        if duplicate_ids:
            raise SegmentationValidationError(
                f"Duplicate object ids before database update: "
                f"{sorted(duplicate_ids)}"
            )
        image_record = {
            # image identification
            "image_id": image_meta["original_key"],
            "clean_key": image_meta["clean_key"],
            # Parsed metadata
            **image_meta,
            # Image data
            "cube": cube,
            "image_ref": image_ref,
            "mask": mask,
            "labels": labels,
            "threshold": tau,
            "segmentation": seg_meta,
            "segmentation_n_labels_positive": seg_meta["n_labels_positive"],
            "segmentation_mask_area_pixels": seg_meta["mask_area_pixels"],
            "segmentation_mask_area_ratio": seg_meta["mask_area_ratio"],
            "segmentation_source": provenance.get("source", "automatic"),
            "segmentation_hash": provenance.get(
                "hash",
                sha256_ndarray(labels),
            ),
            # Spectral metadata
            "wavelengths": wavelengths,
            "data_mode": data_mode,
            # Summary
            "n_objects": len(objects),
            "object_ids": list(objects.keys()),
        }
        for obj in objects.values():
            validate_extracted_object(obj, image_record)
        object_database.update(objects)
        image_database[image_meta["clean_key"]] = image_record
        print(f"  -> {len(objects)} objects detected")

    return object_database, image_database


def is_hyperspectral_cube(value):
    """Return True for values that look like HSI cubes with shape (H, W, B)."""
    return isinstance(value, np.ndarray) and value.ndim == 3


def detect_known_image_keys(data, skip_non_cubes=True):
    """
    Automatically keep images recognized by parse_image_key().

    Recognized examples:
        almond1_sb, peanut2_sb, alm1pea2_sb, pea2_pos1_sb
    """
    rows = []

    for key, value in data.items():
        if skip_non_cubes and not is_hyperspectral_cube(value):
            continue

        meta = parse_image_key(key)

        if meta["is_unknown"]:
            continue

        rows.append((key, meta))

    return rows


def resolve_selected_keys(data, selected_keys):
    """
    Resolve user-provided selected keys.

    Accepts exact raw keys, e.g. almond1_sb,
    and clean keys, e.g. almond1.
    """
    if not selected_keys:
        return None

    raw_keys = set(data.keys())
    clean_to_raw = {}

    for raw_key in data.keys():
        meta = parse_image_key(raw_key)
        if not meta["is_unknown"]:
            clean_to_raw[meta["clean_key"]] = raw_key

    resolved = []
    missing = []

    for key in selected_keys:
        if key in raw_keys:
            resolved.append(key)
            continue

        key_lower = str(key).strip().lower()

        if key_lower in clean_to_raw:
            resolved.append(clean_to_raw[key_lower])
            continue

        key_with_suffix = f"{key_lower}_sb"

        if key_with_suffix in raw_keys:
            resolved.append(key_with_suffix)
            continue

        missing.append(key)

    if missing:
        raise KeyError(
            "Some selected keys were not found in the .mat file: "
            + ", ".join(map(str, missing))
        )

    return resolved


def build_image_summary(image_db) -> pd.DataFrame:
    """Return the compact canonical image manifest without spectral columns."""
    rows = []
    for clean_key, image in image_db.items():
        cube = np.asarray(image.get("cube"))
        mask = np.asarray(image.get("mask"), dtype=bool)
        rows.append(
            {
                "clean_key": str(clean_key),
                "sample_kind": image.get("sample_kind"),
                "nut_type": image.get("nut_type"),
                "batch": image.get("batch"),
                "image_status": image.get("image_status", "accepted"),
                "n_objects": int(image.get("n_objects", 0)),
                "height": int(cube.shape[0]) if cube.ndim == 3 else np.nan,
                "width": int(cube.shape[1]) if cube.ndim == 3 else np.nan,
                "n_bands": int(cube.shape[2]) if cube.ndim == 3 else np.nan,
                "mask_area_ratio": (
                    float(mask.mean()) if mask.size else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=expcfg.DATABASE_IMAGE_SUMMARY_COLUMNS)


def build_object_summary(object_db) -> pd.DataFrame:
    """Return the compact canonical object manifest without spectra."""
    rows = [
        {
            "object_id": str(object_id),
            "source_image": obj.get(
                "source_clean_key",
                obj.get("source_image"),
            ),
            "sample_kind": obj.get("sample_kind"),
            "object_nut_type": obj.get("object_nut_type"),
            "batch": obj.get("batch"),
            "object_status": obj.get("object_status", "accepted"),
            "area_pixels": int(obj.get("area_pixels", 0)),
            "n_pixels": int(obj.get("n_pixels", 0)),
            "n_bands": int(obj.get("n_bands", 0)),
        }
        for object_id, obj in object_db.items()
    ]
    return pd.DataFrame(rows, columns=expcfg.DATABASE_OBJECT_SUMMARY_COLUMNS)
