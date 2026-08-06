from __future__ import annotations

import pandas as pd

from src.utils import save_parquet


def test_save_parquet_ignores_transient_non_json_dataframe_attrs(tmp_path):
    source = pd.DataFrame({"value": [1, 2]})
    source.attrs["diagnostic_table"] = pd.DataFrame(
        {"stage": ["complete"], "n_rows": [2]}
    )

    for optimize in (False, True):
        path = save_parquet(
            source,
            tmp_path / f"attrs_{optimize}.parquet",
            optimize=optimize,
        )
        restored = pd.read_parquet(path)

        pd.testing.assert_frame_equal(restored, source, check_dtype=False)
        assert restored.attrs == {}
