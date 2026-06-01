import re
import numpy as np
from skimage import measure
from copy import deepcopy

from src.segmentation import segment_objects

NIR_UCO_NAME_CONFIG = {
    "suffixes_to_ignore": ["_sb"],
    #ADD NUTS HERE IF NEEDED IN THE FUTURE
    "nut_aliases": {
        "almond": "almond",
        "alm": "almond",
        "peanut": "peanut",
        "pea": "peanut",
        "walnut": "walnut",
        "wal": "walnut",
    },
    #ADD NUTS HERE IF NEEDED IN THE FUTURE
    "patterns": {
        "pure": r"^(?P<nut_token>almond|peanut|walnut)(?P<batch>\d+)$",
        "mixture": r"^(?P<components>(?:alm|pea|wal)\d+){2,}$",
        "position_reference": r"^(?P<nut_token>pea|wal|alm)(?P<batch>\d+)_pos(?P<position_set>\d+)$",
    },
}



def _remove_known_suffixes(name, suffixes):
    clean = name
    for suffix in suffixes:
        if clean.endswith(suffix):
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


def _parse_components(component_string, nut_aliases):
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
    # catch all occurrences of nut token followed by batch number
    pairs = re.findall(r"([a-zA-Z]+)(\d+)", component_string)
    components = {}
    for token, batch in pairs:
        token = token.lower()
        nut_type = nut_aliases.get(token)
        if nut_type is None:
            raise ValueError(f"Unknown nut token: {token}")
        components[nut_type] = {
            "batch": int(batch),
            "token": token,
        }
    return components

def _infer_split_from_metadata(meta):
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

def _infer_object_nut_type_from_metadata(meta):
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


def preprocess_nir_uco_cube(
    raw_cube,
    n_remove:int,
):
    """
    Minimal preprocessing for NIR UCO cubes

    Remove first noisy bands:
        X_clean = X_raw[:, :, 6:]
    Expend Later
    """
    cube = raw_cube[:, :, n_remove:]
    return cube


def parse_image_key(key, config=None):
    """
    Parse image name and return standardised metadata.

    Handle with configurable patterns :
    - pure images : almond1, peanut2
    - mixtures : alm1pea2
    - position references : pea1_pos3
    - suffixes to ignore : _sb

    Parameters
    ----------
    key : str
        Name of the image in the .mat file.
    config : dict or None
        Parsing configuration. If None, uses NIR_UCO_NAME_CONFIG.

    Returns
    -------
    dict
        Standardised metadata.
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
    patterns = config["patterns"]

    # Pure images
    pure_match = re.fullmatch(patterns["pure"], clean_key)
    if pure_match:
        nut_token = pure_match.group("nut_token")
        batch = int(pure_match.group("batch"))
        nut_type = nut_aliases.get(nut_token, nut_token)
        meta.update({
            "sample_kind": "pure",
            "nut_type": nut_type,
            "batch": batch,
            "components": {
                nut_type: {
                    "batch": batch,
                    "token": nut_token,
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
        components = _parse_components(clean_key, nut_aliases)
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

    # Reference positions
    pos_match = re.fullmatch(patterns["position_reference"], clean_key)
    if pos_match:
        nut_token = pos_match.group("nut_token")
        batch = int(pos_match.group("batch"))
        position_set = int(pos_match.group("position_set"))
        nut_type = nut_aliases.get(nut_token, nut_token)
        meta.update({
            "sample_kind": "position_reference",
            "nut_type": nut_type,
            "batch": batch,
            "components": {
                nut_type: {
                    "batch": batch,
                    "token": nut_token,
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
    if split is None:
        split = _infer_split_from_metadata(image_meta)
    object_nut_type = _infer_object_nut_type_from_metadata(image_meta)
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
        # X_k ∈ R^{N_k x B}
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
    n_remove=6,
    wavelengths=None,
    data_mode="reflectance",
    min_area=100,
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
    n_remove : int
        Number of initial bands to remove if using preprocess_nir_uco_cube.
    wavelengths : np.ndarray or None
        Wavelength axis after preprocessing.
    data_mode : str
        "reflectance" or "absorbance".
    min_area : int
        Minimum area for extracted objects.
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
        if preprocess_func is not None:
            cube = preprocess_func(raw_cube, n_remove=n_remove)
        else:
            cube = np.asarray(raw_cube)
        seg_result = segment_objects(
            cube,
            **segmentation_kwargs,
        )

        image_ref, mask, labels, tau = _unpack_segmentation_result(seg_result)
        objects = extract_objects_from_labeled_image(
            cube=cube,
            labels=labels,
            image_ref=image_ref,
            image_meta=image_meta,
            wavelengths=wavelengths,
            data_mode=data_mode,
            min_area=min_area,
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
            # Spectral metadata
            "wavelengths": wavelengths,
            "data_mode": data_mode,
            # Summary
            "n_objects": len(objects),
            "object_ids": list(objects.keys()),
        }
        print(f"  -> {len(objects)} objects detected")

    return object_database, image_database