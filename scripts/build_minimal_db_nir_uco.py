from pathlib import Path
import sys
import pickle
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.io.dataload import load_mat_file
from src.data.database import build_minimal_nir_uco_object_database, preprocess_nir_uco_cube, parse_image_key
from src.utils import wavelength_axis, save_pickle, make_wavelengths
from src.io.database_h5 import save_nir_uco_h5

START_NM = 889
END_NM = 1702
ORIGINAL_BANDS = 69
N_REMOVE = 6
RAW_DATA_PATH = PROJECT_ROOT / "HSI Data" / "NIR camera UCO (889-1702 nm)" / "NIR_uco_sb.mat"
PROCESSED_DATA_DIR = PROJECT_ROOT / "HSI Data" / "processed"
# SELECTED_KEYS = [
#     "almond3_sb",
#     "almond4_sb",
#     "peanut3_sb",
#     "peanut4_sb",
# ]
DATA_MODE = "reflectance"
DEFAULT_SEGMENTATION_KWARGS = {
    "reference_method": "max",
    "threshold_method": "fixed",
    "tau_min": 0.02,
    "opening_radius": 0,
    "closing_radius": 1,
    "fill_holes": True,
    "min_distance": 10,
}

def is_hyperspectral_cube(value):
    """Return True for values that look like HSI cubes with shape (H, W, B)."""
    return isinstance(value, np.ndarray) and value.ndim == 3


def detect_known_image_keys(data, skip_non_cubes=True):
    """
    Automatically keep images recognized by parse_image_key().

    Recognized examples:
        almond1_sb, peanut2_sb, alm1pea2_sb, pea2_pos1_sb

    Unknown names are ignored. If skip_non_cubes=True, non-3D arrays are ignored.
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

    Accepts exact raw keys, e.g. almond1_sb, and clean keys, e.g. almond1.
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


def build_output_paths(output_dir, output_prefix, data_mode):
    """Create output paths for pkl, h5 and wavelength files."""
    output_dir = Path(output_dir)
    stem = f"{output_prefix}_{data_mode}"

    return {
        "object_pkl": output_dir / f"{stem}_object_db.pkl",
        "image_pkl": output_dir / f"{stem}_image_db.pkl",
        "h5": output_dir / f"{stem}.h5",
        "wavelengths": output_dir / f"{stem}_wavelengths.npy",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build NIR UCO object/image databases from .mat files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=RAW_DATA_PATH,
        help="Path to NIR_uco_sb.mat",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help="Directory where databases will be saved",
    )
    parser.add_argument(
        "--format",
        choices=["pkl", "h5", "both"],
        default="h5",
        help="Database output format: pkl, h5, or both.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="nir_uco_all_images",
        help="Prefix used for output filenames.",
    )
    parser.add_argument(
        "--selected-keys",
        nargs="*",
        default=None,
        help=(
            "Optional image keys to process. Accepts raw keys such as almond1_sb "
            "or clean keys such as almond1. If omitted, all recognized images are processed."
        ),
    )
    parser.add_argument(
        "--n-remove",
        type=int,
        default=N_REMOVE,
        help="Number of first noisy wavelength bands to remove.",
    )
    parser.add_argument(
        "--start-nm",
        type=float,
        default=START_NM,
        help="First wavelength of the original raw cube.",
    )
    parser.add_argument(
        "--end-nm",
        type=float,
        default=END_NM,
        help="Last wavelength of the original raw cube.",
    )
    parser.add_argument(
        "--original-bands",
        type=int,
        default=ORIGINAL_BANDS,
        help="Number of wavelength bands in the original raw cube.",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default=DATA_MODE,
        choices=["reflectance", "absorbance"],
        help="Data mode stored in metadata. Raw NIR UCO data are reflectance.",
    )
    parser.add_argument(
        "--use-watershed",
        action="store_true",
        help="Use watershed to split touching objects",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum object area in pixels",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="projection",
        help="Split label for all objects",
    )
    parser.add_argument(
        "--include-heavy-object-arrays",
        action="store_true",
        help=(
            "For HDF5 only: also save mask_global, cube_crop and image_ref_crop "
            "for each object. Without this flag, HDF5 is compact and these arrays "
            "can be reconstructed when loading."
        ),
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}\n\n"
            f"Expected path:\n{RAW_DATA_PATH}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"Input file      : {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output format   : {args.format}")
    print(f"Output prefix   : {args.output_prefix}")
    print(f"Data mode       : {args.data_mode}")
    print(f"n_remove        : {args.n_remove}")
    print(f"Use watershed   : {args.use_watershed}")
    print(f"Minimum area    : {args.min_area}")
    print(f"Forced split    : {args.split if args.split is not None else 'None (auto inferred)'}")
    print("=" * 80)

    data = load_mat_file(input_path)

    if args.selected_keys:
        selected_keys = resolve_selected_keys(data, args.selected_keys)
        detected_rows = [(key, parse_image_key(key)) for key in selected_keys]
        print(f"Selected images : {len(selected_keys)} user-selected image(s)")
    else:
        detected_rows = detect_known_image_keys(data, skip_non_cubes=True)
        selected_keys = [key for key, _ in detected_rows]
        print("Selected images : all automatically recognized images")
    if not selected_keys:
        raise RuntimeError(
            "No recognized NIR UCO image was found. Check image names and parsing patterns."
        )
    
    wavelengths = make_wavelengths(
            start_nm=args.start_nm,
            end_nm=args.end_nm,
            original_bands=args.original_bands,
            n_remove=args.n_remove,
        )

    segmentation_kwargs = dict(DEFAULT_SEGMENTATION_KWARGS)
    segmentation_kwargs.update(
        {
            "min_area": args.min_area,
            "use_watershed": args.use_watershed,
        }
    )

    object_db, image_db = build_minimal_nir_uco_object_database(
        data=data,
        selected_keys=selected_keys,
        preprocess_func=preprocess_nir_uco_cube,
        n_remove=args.n_remove,
        wavelengths=wavelengths,
        data_mode=args.data_mode,
        min_area=args.min_area,
        split=args.split,
        skip_unknown=True,
        segmentation_kwargs=segmentation_kwargs,
    )
    paths = build_output_paths(
        output_dir=output_dir,
        output_prefix=args.output_prefix,
        data_mode=args.data_mode,
    )
    saved_paths = []
    if args.format in {"pkl", "both"}:
        save_pickle(object_db, paths["object_pkl"])
        save_pickle(image_db, paths["image_pkl"])
        saved_paths.extend([paths["object_pkl"], paths["image_pkl"]])
    if args.format in {"h5", "both"}:
        save_nir_uco_h5(
            object_database=object_db,
            image_database=image_db,
            path=paths["h5"],
            include_heavy_object_arrays=args.include_heavy_object_arrays,
        )
        saved_paths.append(paths["h5"])
    np.save(paths["wavelengths"], wavelengths)
    saved_paths.append(paths["wavelengths"])
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Number of images processed : {len(image_db)}")
    print(f"Number of objects extracted: {len(object_db)}")
    print(f"Wavelength bands kept      : {len(wavelengths)}")
    print("Saved files:")
    for path in saved_paths:
        print(f"  [OK] {path}")
    print("=" * 80)
    print("Images")
    print("=" * 80)
    for image_id, image_obj in image_db.items():
        print(
            f"{image_id:15s} | "
            f"kind={image_obj['sample_kind']:18s} | "
            f"objects={image_obj['n_objects']}"
        )
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()