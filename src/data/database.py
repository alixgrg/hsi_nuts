from copy import deepcopy
import re

import numpy as np
from skimage import measure

from src.data.segmentation import segment_objects

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
        )
    if isinstance(seg_result, tuple) and len(seg_result) == 4:
        image_ref, mask, labels, tau = seg_result
        return image_ref, mask, labels, tau
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
            "description": f"mixture: {component_desc}",
        })
        return meta

    return meta



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
    obj_counter = 1

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
        object_id = f"{image_meta['clean_key']}_obj{obj_counter:03d}"
        objects[object_id] = {
            # object identification
            "object_id": object_id,
            "object_index": obj_counter,
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
        obj_counter += 1

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
    skip_unknown=True,
    segmentation_kwargs=None,
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
        seg_result = segment_objects(
            cube,
            **segmentation_kwargs,
        )

        image_ref, mask, labels, tau = _unpack_segmentation_result(seg_result)
        _validate_cube_and_segmentation(cube, labels, image_ref, mask=mask)
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
        object_database.update(objects)
        image_database[image_meta["clean_key"]] = {
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
            # Spectral metadata
            "wavelengths": wavelengths,
            "data_mode": data_mode,
            # Summary
            "n_objects": len(objects),
            "object_ids": list(objects.keys()),
        }
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
