"""Materialize the versioned historical QC decisions outside configuration."""

from pathlib import Path

import pandas as pd


RECORD_IDS = (
    "alm1pea1_obj030",
    "alm3pea3_obj005",
    "alm3pea3_obj029",
    "pea3_pos3_obj004",
    "pea3_pos3_obj008",
    "almond2_obj032",
    "alm3pea2_obj026",
    "alm3pea4_obj008",
)
FLAGGED_RECORDS = {
    "possible_merged_object": [
        "alm1pea1_obj030",
        "alm3pea3_obj005",
        "alm3pea3_obj029",
        "pea3_pos3_obj004",
        "pea3_pos3_obj008",
        "almond2_obj032",
    ],
    "robust_spectral_outlier": ["alm3pea2_obj026", "alm3pea4_obj008"],
}

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "docs" / "protocol" / "8tracks_v4" / "qc_review_decisions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "record_type": "object",
            "record_id": record_id,
            "flag_type": "possible_merged_object",
            "review_status": "reviewed",
            "review_decision": "accept_as_is",
            "reviewer": "visual_review",
            "review_date": "2026-08-07",
            "review_comment": (
                "Visual inspection of the source image, object mask and "
                "spectrum found no defensible segmentation correction."
            ),
            "review_evidence": "qc_visual_review_report.pdf",
        }
        for record_id in FLAGGED_RECORDS["possible_merged_object"]
    ]
    rows += [
        {
            "record_type": "object",
            "record_id": record_id,
            "flag_type": "robust_spectral_outlier",
            "review_status": "reviewed",
            "review_decision": "accept_as_is",
            "reviewer": "visual_review",
            "review_date": "2026-08-07",
            "review_comment": (
                "Ignoring for now"
            ),
            "review_evidence": "qc_visual_review_report.pdf",
        }
        for record_id in FLAGGED_RECORDS["robust_spectral_outlier"]
    ]
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
