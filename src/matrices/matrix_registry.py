from __future__ import annotations

from dataclasses import dataclass

from src.matrices.redim_matrix import object_db_to_matrix


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    level: str
    spectrum_field: str = "mean_spectrum"
    description: str = ""
    uses_pixel_sampling: bool = False


MATRIX_REGISTRY: dict[str, MatrixSpec] = {
    "object_mean": MatrixSpec(
        name="object_mean",
        level="object",
        spectrum_field="mean_spectrum",
        description="One observation per object using the mean spectrum.",
        uses_pixel_sampling=False,
    ),
    "object_median": MatrixSpec(
        name="object_median",
        level="object",
        spectrum_field="median_spectrum",
        description="One observation per object using the median spectrum.",
        uses_pixel_sampling=False,
    ),
    "balanced_pixels": MatrixSpec(
        name="balanced_pixels",
        level="balanced_pixel",
        spectrum_field="mean_spectrum",
        description="m pixels sampled per object.",
        uses_pixel_sampling=True,
    ),
    "all_pixels": MatrixSpec(
        name="all_pixels",
        level="pixel",
        spectrum_field="mean_spectrum",
        description="All object pixels as observations.",
        uses_pixel_sampling=False,
    ),
    "pixel": MatrixSpec(
        name="pixel",
        level="pixel",
        spectrum_field="mean_spectrum",
        description="Alias for all_pixels.",
        uses_pixel_sampling=False,
    ),
}


def available_matrix_methods() -> list[str]:
    """Return available matrix method names."""
    return sorted(MATRIX_REGISTRY.keys())


def get_matrix_spec(matrix_method: str) -> MatrixSpec:
    """Return the MatrixSpec associated with a matrix method."""
    matrix_method = str(matrix_method)

    if matrix_method not in MATRIX_REGISTRY:
        raise ValueError(
            f"Unknown matrix_method={matrix_method!r}. "
            f"Available methods are: {available_matrix_methods()}"
        )

    return MATRIX_REGISTRY[matrix_method]


def matrix_method_to_args(matrix_method: str) -> dict:
    """
    Compatibility helper.

    Returns arguments expected by object_db_to_matrix.
    """
    spec = get_matrix_spec(matrix_method)

    return {
        "level": spec.level,
        "spectrum_field": spec.spectrum_field,
    }


def build_matrix(
    object_db,
    matrix_method: str,
    filters: dict | None = None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    balanced_pixel_strategy: str = "random",
):
    """
    Build X, y, metadata from object_db using a registered matrix method.

    Parameters
    ----------
    matrix_method:
        "object_mean", "object_median", "balanced_pixels", "all_pixels" or "pixel".

        Used only for matrix_method="balanced_pixels".
        Recommended values after redim_matrix.py update:
        - "random"
        - "center"
        - "core_random"
    """
    spec = get_matrix_spec(matrix_method)

    return object_db_to_matrix(
        object_db=object_db,
        level=spec.level,
        spectrum_field=spec.spectrum_field,
        filters=filters or {},
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )