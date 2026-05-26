from pathlib import Path
import sys
import pickle
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import load_mat_file
from src import build_minimal_nir_uco_object_database

START_NM = 889
END_NM = 1702
ORIGINAL_BANDS = 69
N_REMOVE = 6
RAW_DATA_PATH = PROJECT_ROOT / "HSI Data" / "NIR camera UCO (889-1702 nm)" / "NIR_uco_sb.mat"
PROCESSED_DATA_DIR = PROJECT_ROOT / "HSI Data" / "processed"
SELECTED_KEYS = [
    "almond1_sb",
    "almond2_sb",
    "peanut1_sb",
    "peanut2_sb",
]
DATA_MODE = "reflectance"

def preprocess_nir_uco_cube(
    raw_cube,
    n_remove=N_REMOVE,
):
    """
    Minimal preprocessing for NIR UCO cubes

    Remove first noisy bands:
        X_clean = X_raw[:, :, 6:]
    Expend Later
    """
    cube = raw_cube[:, :, n_remove:]
    return cube

def make_wavelength_axis(
    start_nm=START_NM,
    end_nm=END_NM,
    original_bands=ORIGINAL_BANDS,
    n_remove=N_REMOVE,
):
    """
    Approximate wavelength axis after removing noisy bands.

    The raw cube has 69 bands between 889 and 1702 nm.
    The first 6 bands are removed.

    Returns:
        wavelengths ∈ R^{63}
    """
    wavelengths_full = np.linspace(start_nm, end_nm, original_bands)
    return wavelengths_full[n_remove:]

def save_pickle(obj, path):
    """
    Save Python object as pickle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[OK] Saved: {path}")




def main():
    parser = argparse.ArgumentParser(
        description="Build minimal NIR UCO object database."
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
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}\n\n"
            f"Expected path:\n{RAW_DATA_PATH}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = SELECTED_KEYS
    print("=" * 80)
    print("Building minimal NIR UCO object database")
    print("=" * 80)
    print(f"Input file      : {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Selected images : {selected_keys}")
    print(f"Data mode       : {DATA_MODE}")
    print(f"Use watershed   : {args.use_watershed}")
    print(f"Minimum area    : {args.min_area}")
    print("=" * 80)

    data = load_mat_file(input_path)
    wavelengths = make_wavelength_axis()

    def preprocess_func(raw_cube):
        return preprocess_nir_uco_cube(raw_cube)

    object_db, image_db = build_minimal_nir_uco_object_database(
        data=data,
        selected_keys=selected_keys,
        preprocess_func=preprocess_func,
        wavelengths=wavelengths,
        data_mode=DATA_MODE,
        min_area=args.min_area,
        skip_unknown=True,
        segmentation_kwargs={
            "reference_method": "max",
            "threshold_method": "fixed",
            "tau_min": 0.02,
            "min_area": args.min_area,
            "opening_radius": 0,
            "closing_radius": 1,
            "fill_holes": True,
            "use_watershed": args.use_watershed,
            "min_distance": 10,
        },
    )
    object_db_path = output_dir / f"nir_uco_minimal_object_db_{DATA_MODE}.pkl"
    image_db_path = output_dir / f"nir_uco_minimal_image_db_{DATA_MODE}.pkl"
    wavelengths_path = output_dir / f"nir_uco_wavelengths_{DATA_MODE}.npy"
    save_pickle(object_db, object_db_path)
    save_pickle(image_db, image_db_path)
    np.save(wavelengths_path, wavelengths)

    print(f"[OK] Saved: {wavelengths_path}")
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Number of images processed : {len(image_db)}")
    print(f"Number of objects extracted: {len(object_db)}")

    for image_id, image_obj in image_db.items():
        print(
            f"{image_id:12s} | "
            f"kind={image_obj['sample_kind']:8s} | "
            f"objects={image_obj['n_objects']}"
        )
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()