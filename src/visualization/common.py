from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from src.utils import as_1d_array

CLASS_COLOR_MAP: dict[str, str] = {
    "peanut": "limegreen",
    "target": "limegreen",
    "peanut_only": "limegreen",
    "almond": "royalblue",
    "non_target": "royalblue",
    "non_peanut": "royalblue",
    "almond_only": "royalblue",
    "uncertain": "purple",
    "ambiguous": "purple",
    "unknown": "lightgray",
    "missing": "lightgray",
}

ERROR_COLOR_MAP: dict[str, str] = {
    "TP": "limegreen",
    "TN": "royalblue",
    "FP": "orange",
    "FN": "red",
    "unavailable": "lightgray",
}

BINARY_CLASS_ORDER = ("almond", "peanut")
THREE_WAY_CLASS_ORDER = ("almond", "uncertain", "peanut")
ERROR_ORDER = ("TP", "TN", "FP", "FN")

_NAMED_RGB = {
    "limegreen": (50, 205, 50),
    "royalblue": (65, 105, 225),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "red": (255, 0, 0),
    "crimson": (220, 20, 60),
    "lightgray": (211, 211, 211),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}



def normalize_class_label(
    value: Any,
    *,
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
    unknown_label: str = "unknown",
) -> str:
    """Map project label aliases to one stable display label.

    Unknown values are preserved as strings instead of silently returning None.
    """
    if value is None:
        return "missing"
    try:
        if pd.isna(value):
            return "missing"
    except Exception:
        pass

    raw = str(value).strip()
    val = raw.lower()

    target_aliases = {
        str(target_class).lower(),
        "target",
        "peanut",
        "peanut_only",
        "positive",
        "true",
        "1",
    }
    non_target_aliases = {
        str(non_target_label).lower(),
        "non_target",
        "non-target",
        "non_peanut",
        "almond",
        "almond_only",
        "negative",
        "false",
        "0",
    }
    uncertain_aliases = {
        str(uncertain_label).lower(),
        "uncertain",
        "ambiguous",
        "indeterminate",
    }
    unknown_aliases = {"unknown", "unavailable", "missing", "nan", "none", ""}

    if val in target_aliases:
        return str(target_class)
    if val in non_target_aliases:
        return str(non_target_label)
    if val in uncertain_aliases:
        return str(uncertain_label)
    if val in unknown_aliases:
        return str(unknown_label)
    return raw


def normalize_class_array(
    values,
    *,
    target_class: str = "peanut",
    non_target_label: str = "almond",
    uncertain_label: str = "uncertain",
) -> np.ndarray:
    return np.asarray(
        [
            normalize_class_label(
                value,
                target_class=target_class,
                non_target_label=non_target_label,
                uncertain_label=uncertain_label,
            )
            for value in values
        ],
        dtype=str,
    )


def class_color(label: Any, default: str = "lightgray") -> str:
    normalized = normalize_class_label(label)
    return CLASS_COLOR_MAP.get(normalized.lower(), CLASS_COLOR_MAP.get(str(label).lower(), default))


def class_color_map(
    labels: Sequence[Any] | None = None,
    *,
    include_unknown: bool = False,
) -> dict[str, str]:
    """Return a stable colour map, optionally restricted to supplied labels."""
    if labels is None:
        keys = ["almond", "uncertain", "peanut"]
        if include_unknown:
            keys.append("unknown")
        return {key: class_color(key) for key in keys}

    out: dict[str, str] = {}
    for label in ordered_unique(labels):
        out[str(label)] = class_color(label)
    return out


def color_with_alpha(color: str, alpha: float) -> str:
    """Convert a named or hex colour to an rgba string."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    key = str(color).lower()
    if key in _NAMED_RGB:
        rgb = _NAMED_RGB[key]
    elif key.startswith("#"):
        rgb = pc.hex_to_rgb(key)
    elif key.startswith("rgb("):
        values = key.removeprefix("rgb(").removesuffix(")").split(",")
        rgb = tuple(int(float(v.strip())) for v in values[:3])
    else:
        # Safe fallback for uncommon CSS names.
        rgb = _NAMED_RGB["lightgray"]
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.4f})"



def show_or_return(fig: go.Figure, show: bool = True):
    """Show a Plotly figure or return it."""
    if show:
        fig.show()
        return None
    return fig


def make_customdata(n: int, **metadata) -> tuple[np.ndarray, str]:
    """
    Build Plotly customdata and hovertemplate block from metadata arrays.

    Each metadata value is broadcast/validated with as_1d_array.
    """
    names = [k for k, v in metadata.items() if v is not None]
    if not names:
        return np.empty((n, 0), dtype=str), ""
    cols = [
        as_1d_array(metadata[k], n, "").astype(str)
        for k in names
    ]
    data = np.stack(cols, axis=1)
    hover = "".join(
        f"{name}: %{{customdata[{i}]}}<br>"
        for i, name in enumerate(names)
    )
    return data, hover


def ordered_unique(values) -> list[str]:
    """Return unique values while preserving first-seen order."""
    return list(dict.fromkeys(np.asarray(values).astype(str)))


def make_dynamic_color_map(
    groups,
    color_sequence=None,
    continuous_colorscale: str = "Turbo",
    prefer_project_colors: bool = True,
) -> dict[str, str]:
    """Build a categorical colour map for any number of groups.

    Known project classes retain their stable colours; all other groups receive a
    deterministic Plotly categorical colour.
    """
    unique_groups = ordered_unique(groups)

    out: dict[str, str] = {}
    remaining: list[str] = []
    for group in unique_groups:
        normalized = normalize_class_label(group)
        if prefer_project_colors and normalized.lower() in CLASS_COLOR_MAP:
            out[group] = CLASS_COLOR_MAP[normalized.lower()]
        else:
            remaining.append(group)

    if color_sequence is None:
        color_sequence = (
            pc.qualitative.Plotly
            + pc.qualitative.D3
            + pc.qualitative.G10
            + pc.qualitative.T10
            + pc.qualitative.Alphabet
        )

    if len(remaining) <= len(color_sequence):
        colors = color_sequence[: len(remaining)]
    else:
        colors = pc.sample_colorscale(
            continuous_colorscale,
            np.linspace(0, 1, len(remaining)),
        )

    for index, group in enumerate(remaining):
        out[group] = colors[index]
    return out


def discrete_colorscale(colors: Sequence[str]) -> list[list[object]]:
    """Build a stepwise Plotly colorscale for categorical integer codes."""
    colors = list(colors)
    if not colors:
        raise ValueError("At least one colour is required.")
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]

    scale: list[list[object]] = []
    n = len(colors)
    for index, color in enumerate(colors):
        left = index / n
        right = (index + 1) / n
        scale.append([left, color])
        scale.append([right, color])
    return scale


def apply_project_theme(
    fig: go.Figure,
    *,
    template: str = "plotly_white",
    font_size: int = 13,
    title_font_size: int = 18,
    legend_orientation: str | None = None,
) -> go.Figure:
    """Apply a restrained, report-ready layout without changing trace colours."""
    layout_updates: dict[str, Any] = {
        "template": template,
        "font": {"size": font_size},
        "title": {"font": {"size": title_font_size}, "x": 0.02},
        "margin": {"l": 70, "r": 80, "t": 75, "b": 65},
        "hoverlabel": {"namelength": -1},
    }
    if legend_orientation is not None:
        layout_updates["legend"] = {
            "orientation": legend_orientation,
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        }
    fig.update_layout(**layout_updates)
    return fig

def background_image(
    image_db: Mapping[str, Mapping[str, Any]],
    image_key: str,
    base: str = "image_ref",
    band: int | None = None,
):
    """
    Return a 2D background image from image_db.

    base:
        - "image_ref"
        - any 2D key in image_db[image_key]
        - "band" to extract one cube band
    """
    if image_key not in image_db:
        raise KeyError(f"Image not found in image_db: {image_key}")

    img = image_db[image_key]

    if base == "band":
        if "cube" not in img:
            raise KeyError(f"Image {image_key!r} has no 'cube' field.")
        if band is None:
            band = img["cube"].shape[2] // 2
        return img["cube"][:, :, int(band)]

    if base in img:
        return img[base]
    if "image_ref" in img:
        return img["image_ref"]

    raise KeyError(
        f"Could not find base={base!r} or fallback 'image_ref' "
        f"for image {image_key!r}."
    )


def validate_columns(df, columns, df_name: str = "df"):
    """Raise a clear error if required columns are missing."""
    missing = [col for col in columns if col is not None and col not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {df_name}: {missing}")


def foreground_bbox(mask: np.ndarray, padding: int = 0) -> tuple[int, int, int, int] | None:
    """Return ``(row_start, row_stop, col_start, col_stop)`` around a 2D mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    padding = max(int(padding), 0)
    r0 = max(int(rows.min()) - padding, 0)
    r1 = min(int(rows.max()) + padding + 1, mask.shape[0])
    c0 = max(int(cols.min()) - padding, 0)
    c1 = min(int(cols.max()) + padding + 1, mask.shape[1])
    return r0, r1, c0, c1


def crop_to_foreground(
    array: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    padding: int = 0,
    background_value: float | int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a 2D array to a foreground mask and return the crop plus its bbox."""
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError("array must be 2D.")

    if mask is None:
        if np.issubdtype(arr.dtype, np.floating):
            mask = np.isfinite(arr) & (arr != background_value)
        else:
            mask = arr != background_value
    bbox = foreground_bbox(np.asarray(mask, dtype=bool), padding=padding)
    if bbox is None:
        return arr.copy(), (0, arr.shape[0], 0, arr.shape[1])
    r0, r1, c0, c1 = bbox
    return arr[r0:r1, c0:c1], bbox


def crop_arrays_to_foreground(
    arrays: Sequence[np.ndarray],
    mask: np.ndarray,
    *,
    padding: int = 0,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    """Crop several equally shaped 2D arrays with one shared foreground bbox."""
    arrays = [np.asarray(array) for array in arrays]
    if not arrays:
        raise ValueError("arrays cannot be empty.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("All arrays must have the same shape.")

    bbox = foreground_bbox(mask, padding=padding)
    if bbox is None:
        return [array.copy() for array in arrays], (0, shape[0], 0, shape[1])
    r0, r1, c0, c1 = bbox
    return [array[r0:r1, c0:c1] for array in arrays], bbox


def sanitize_filename(value: Any, replacement: str = "_") -> str:
    """Create a stable filesystem-safe filename fragment."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", replacement, text)
    text = re.sub(rf"{re.escape(replacement)}+", replacement, text)
    return text.strip("._-") or "unnamed"


def make_config_display_name(
    row: Mapping[str, Any] | pd.Series,
    *,
    fields: Sequence[str] = (
        "selected_config_id",
        "matrix_family",
        "matrix_method",
        "preprocessing",
        "rule",
        "n_components",
    ),
    separator: str = "__",
) -> str:
    """Build a compact deterministic name from available configuration fields."""
    parts: list[str] = []
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        parts.append(f"{sanitize_filename(field)}-{sanitize_filename(value)}")
    return separator.join(parts) if parts else "configuration"


def save_figure_bundle(
    fig: go.Figure,
    output_stem: str | Path,
    *,
    formats: Sequence[str] = ("html", "png"),
    width: int | None = None,
    height: int | None = None,
    scale: float = 2.0,
    include_plotlyjs: str | bool = "cdn",
    strict: bool = False,
) -> dict[str, Path]:
    """Save one Plotly figure in several formats.

    Static image export requires Kaleido. When ``strict=False``, unavailable static
    exporters are skipped while HTML export still succeeds.
    """
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if stem.suffix:
        stem = stem.with_suffix("")

    saved: dict[str, Path] = {}
    for raw_format in formats:
        fmt = str(raw_format).lower().lstrip(".")
        path = stem.with_suffix(f".{fmt}")
        try:
            if fmt == "html":
                fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
            elif fmt in {"png", "jpg", "jpeg", "svg", "pdf", "webp"}:
                fig.write_image(
                    path,
                    width=width,
                    height=height,
                    scale=scale,
                )
            elif fmt == "json":
                path.write_text(fig.to_json(), encoding="utf-8")
            else:
                raise ValueError(f"Unsupported figure format: {fmt}")
            saved[fmt] = path
        except Exception:
            if strict:
                raise
    return saved
