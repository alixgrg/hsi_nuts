from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.matrices.redim_matrix import object_db_to_matrix


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    level: str
    spectrum_field: str = "mean_spectrum"
    description: str = ""
    uses_pixel_sampling: bool = False


@dataclass(frozen=True)
class MatrixOutput:
    """Formal matrix-construction contract."""

    X: np.ndarray
    y: np.ndarray
    metadata: dict
    wavelengths: np.ndarray | None
    matrix_method: str
    matrix_spec: MatrixSpec

    def validate(self) -> "MatrixOutput":
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={self.X.shape}")
        if len(self.y) != self.X.shape[0]:
            raise ValueError(f"y length {len(self.y)} does not match X rows {self.X.shape[0]}")
        for key, values in self.metadata.items():
            if len(np.asarray(values)) != self.X.shape[0]:
                raise ValueError(
                    f"metadata[{key!r}] length {len(np.asarray(values))} "
                    f"does not match X rows {self.X.shape[0]}"
                )
        if self.wavelengths is not None and len(self.wavelengths) != self.X.shape[1]:
            raise ValueError(
                f"wavelengths length {len(self.wavelengths)} does not match X columns {self.X.shape[1]}"
            )
        return self

    def as_tuple(self, include_wavelengths: bool = True):
        if include_wavelengths:
            return self.X, self.y, self.metadata, self.wavelengths
        return self.X, self.y, self.metadata


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


def _extract_wavelengths(object_db, metadata):
    object_ids = metadata.get("object_id")
    if object_ids is None:
        return None

    wavelengths_ref = None
    for object_id in dict.fromkeys(np.asarray(object_ids).astype(str)):
        obj = object_db.get(object_id)
        if obj is None or obj.get("wavelengths") is None:
            continue
        wavelengths = np.asarray(obj["wavelengths"], dtype=float)
        if wavelengths_ref is None:
            wavelengths_ref = wavelengths
            continue
        if wavelengths.shape != wavelengths_ref.shape or not np.allclose(
            wavelengths,
            wavelengths_ref,
            equal_nan=True,
        ):
            raise ValueError("Selected objects have inconsistent wavelength axes.")
    return wavelengths_ref


def build_matrix_output(
    object_db,
    matrix_method: str,
    filters: dict | None = None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    balanced_pixel_strategy: str = "random",
):
    """Build a MatrixOutput object from object_db using a registered matrix method."""
    spec = get_matrix_spec(matrix_method)
    X, y, metadata = object_db_to_matrix(
        object_db=object_db,
        level=spec.level,
        spectrum_field=spec.spectrum_field,
        filters=filters or {},
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    output = MatrixOutput(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y),
        metadata={key: np.asarray(value) for key, value in dict(metadata).items()},
        wavelengths=_extract_wavelengths(object_db, metadata),
        matrix_method=str(matrix_method),
        matrix_spec=spec,
    )
    return output.validate()


def build_matrix(
    object_db,
    matrix_method: str,
    filters: dict | None = None,
    m: int = 40,
    random_state: int = 42,
    replace: bool = False,
    balanced_pixel_strategy: str = "random",
    return_wavelengths: bool = False,
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
    return_wavelengths:
        If True, return ``X, y, metadata, wavelengths``. The default keeps
        backward compatibility with existing notebooks: ``X, y, metadata``.
    """
    output = build_matrix_output(
        object_db=object_db,
        matrix_method=matrix_method,
        filters=filters,
        m=m,
        random_state=random_state,
        replace=replace,
        balanced_pixel_strategy=balanced_pixel_strategy,
    )
    return output.as_tuple(include_wavelengths=return_wavelengths)
