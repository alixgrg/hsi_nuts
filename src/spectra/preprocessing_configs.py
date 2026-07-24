from __future__ import annotations

from collections.abc import Mapping, Sequence

VALID_PREPROCESSING_STEPS = {
    "raw",
    "absorbance",
    "snv",
    "msc",
    "vector_norm",
    "sg_smooth",
    "sg_d1",
    "sg_d2",
}

PREPROCESSING_ALIASES: dict[str, tuple[str, ...]] = {
    "raw": ("raw",),
    "absorbance": ("absorbance",),
    "snv": ("snv",),
    "msc": ("msc",),
    "vector_norm": ("vector_norm",),
    "sg_smooth": ("sg_smooth",),
    "sg_d1": ("sg_d1",),
    "sg_d2": ("sg_d2",),

    "absorbance_snv": ("absorbance", "snv"),
    "absorbance_msc": ("absorbance", "msc"),
    "absorbance_sg_smooth": ("absorbance", "sg_smooth"),
    "absorbance_sg_d1": ("absorbance", "sg_d1"),
    "absorbance_sg_d2": ("absorbance", "sg_d2"),

    "snv_sg_smooth": ("snv", "sg_smooth"),
    "snv_sg_d1": ("snv", "sg_d1"),
    "snv_sg_d2": ("snv", "sg_d2"),

    "absorbance_snv_sg_smooth": ("absorbance", "snv", "sg_smooth"),
    "absorbance_snv_sg_d1": ("absorbance", "snv", "sg_d1"),
    "absorbance_snv_sg_d2": ("absorbance", "snv", "sg_d2"),
}


DEFAULT_PREPROCESSING_CONFIGS: dict[str, tuple[str, ...]] = {
    "raw": PREPROCESSING_ALIASES["raw"],
    "absorbance": PREPROCESSING_ALIASES["absorbance"],
    "snv": PREPROCESSING_ALIASES["snv"],
    "msc": PREPROCESSING_ALIASES["msc"],
    "sg_smooth": PREPROCESSING_ALIASES["sg_smooth"],
    "sg_d1": PREPROCESSING_ALIASES["sg_d1"],
    "sg_d2": PREPROCESSING_ALIASES["sg_d2"],
    "vector_norm": PREPROCESSING_ALIASES["vector_norm"],

    "absorbance_snv": PREPROCESSING_ALIASES["absorbance_snv"],
    "absorbance_msc": PREPROCESSING_ALIASES["absorbance_msc"],
    "absorbance_sg_smooth": PREPROCESSING_ALIASES["absorbance_sg_smooth"],
    "absorbance_sg_d1": PREPROCESSING_ALIASES["absorbance_sg_d1"],
    "absorbance_sg_d2": PREPROCESSING_ALIASES["absorbance_sg_d2"],

    "snv_sg_smooth": PREPROCESSING_ALIASES["snv_sg_smooth"],
    "snv_sg_d1": PREPROCESSING_ALIASES["snv_sg_d1"],
    "snv_sg_d2": PREPROCESSING_ALIASES["snv_sg_d2"],

    "absorbance_snv_sg_smooth": PREPROCESSING_ALIASES["absorbance_snv_sg_smooth"],
    "absorbance_snv_sg_d1": PREPROCESSING_ALIASES["absorbance_snv_sg_d1"],
    "absorbance_snv_sg_d2": PREPROCESSING_ALIASES["absorbance_snv_sg_d2"],
}


SIMCA_SEARCH_PREPROCESSING_CONFIGS: dict[str, tuple[str, ...]] = {
    "snv": PREPROCESSING_ALIASES["snv"],
    "absorbance": PREPROCESSING_ALIASES["absorbance"],
    "vector_norm": PREPROCESSING_ALIASES["vector_norm"],
    "absorbance_snv": PREPROCESSING_ALIASES["absorbance_snv"],
    "absorbance_sg_smooth": PREPROCESSING_ALIASES["absorbance_sg_smooth"],
    "absorbance_sg_d1": PREPROCESSING_ALIASES["absorbance_sg_d1"],
    "absorbance_sg_d2": PREPROCESSING_ALIASES["absorbance_sg_d2"],
    "snv_sg_smooth": PREPROCESSING_ALIASES["snv_sg_smooth"],
    "snv_sg_d1": PREPROCESSING_ALIASES["snv_sg_d1"],
    "snv_sg_d2": PREPROCESSING_ALIASES["snv_sg_d2"],
    "absorbance_snv_sg_smooth": PREPROCESSING_ALIASES["absorbance_snv_sg_smooth"],
    "absorbance_snv_sg_d1": PREPROCESSING_ALIASES["absorbance_snv_sg_d1"],
    "absorbance_snv_sg_d2": PREPROCESSING_ALIASES["absorbance_snv_sg_d2"],
}


def preprocessing_name_from_steps(steps: Sequence[str]) -> str:
    """Return a readable name from a sequence of preprocessing steps."""
    steps = validate_preprocessing_steps(steps)
    if steps == ("raw",) or len(steps) == 0:
        return "raw"
    return "_".join(str(step) for step in steps)


def validate_preprocessing_steps(steps: Sequence[str]) -> tuple[str, ...]:
    """Validate and return preprocessing steps as tuple."""
    steps = tuple(steps)

    if len(steps) == 0:
        return ("raw",)

    unknown = [step for step in steps if step not in VALID_PREPROCESSING_STEPS]
    if unknown:
        raise ValueError(
            f"Unknown preprocessing step(s): {unknown}. "
            f"Valid steps are: {sorted(VALID_PREPROCESSING_STEPS)}"
        )

    if "raw" in steps and len(steps) > 1:
        raise ValueError("'raw' must be used alone, not inside a preprocessing chain.")

    return steps


def resolve_preprocessing_steps(item) -> tuple[str, ...]:
    """
    Convert a preprocessing alias or explicit sequence into executable steps.

    Examples
    --------
    "absorbance_sg_d1" -> ("absorbance", "sg_d1")
    ("absorbance", "snv") -> ("absorbance", "snv")
    """
    if isinstance(item, str):
        steps = PREPROCESSING_ALIASES.get(item, (item,))
    else:
        steps = tuple(item)

    return validate_preprocessing_steps(steps)


def normalize_preprocessing_configs(
    configs=None,
    default_configs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, tuple[str, ...]]:
    """
    Normalize preprocessing configurations.

    Accepted inputs
    ---------------
    None:
        returns DEFAULT_PREPROCESSING_CONFIGS.

    dict:
        {"name": ("absorbance", "snv"), ...}

    list/tuple of strings:
        ["raw", "absorbance_snv"]

    list/tuple of tuples/lists:
        [("absorbance", "snv"), ("absorbance", "sg_d1")]
    """
    if default_configs is None:
        default_configs = DEFAULT_PREPROCESSING_CONFIGS

    if configs is None:
        return {
            str(name): validate_preprocessing_steps(steps)
            for name, steps in default_configs.items()
        }

    if isinstance(configs, Mapping):
        out = {}
        for name, steps in configs.items():
            out[str(name)] = resolve_preprocessing_steps(steps)
        return out

    out = {}
    for item in configs:
        steps = resolve_preprocessing_steps(item)
        name = item if isinstance(item, str) else preprocessing_name_from_steps(steps)
        if str(name) in out and out[str(name)] != steps:
            raise ValueError(
                f"Duplicate preprocessing name with different steps: {name!r}"
            )
        out[str(name)] = steps

    return out
