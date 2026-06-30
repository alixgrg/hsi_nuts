from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc

from src.utils import as_1d_array


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
) -> dict[str, str]:
    """
    Build a dynamic color map for any number of categorical groups.
    """
    unique_groups = ordered_unique(groups)
    n_groups = len(unique_groups)

    if color_sequence is None:
        color_sequence = (
            pc.qualitative.Plotly
            + pc.qualitative.D3
            + pc.qualitative.G10
            + pc.qualitative.T10
            + pc.qualitative.Alphabet
        )

    if n_groups <= len(color_sequence):
        colors = color_sequence[:n_groups]
    else:
        colors = pc.sample_colorscale(
            continuous_colorscale,
            np.linspace(0, 1, n_groups),
        )

    return {
        group: colors[i]
        for i, group in enumerate(unique_groups)
    }


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