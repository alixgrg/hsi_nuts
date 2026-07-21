import pandas as pd

from src.workflows.simca_optuna import optuna_trials_to_candidate_configs


def test_optuna_trials_to_candidate_configs_accepts_multiobjective_trials():
    trials = pd.DataFrame(
        [
            {
                "number": 1,
                "state": "COMPLETE",
                "value_0": 0.0,
                "value_1": 0.2,
                "value_2": 0.9,
                "matrix_method": "balanced_pixels",
                "matrix_family": "pixel_matrix",
                "preprocessing": "snv_sg_d1",
                "preprocessing_steps": "snv+sg_d1",
                "rule_variant": "simple_emp_cv",
                "n_components": 5,
                "alpha": 0.01,
                "m": 40,
                "balanced_pixel_strategy": "random",
                "sg_window_length": 11,
                "sg_polyorder": 2,
                "position_dilation_radius": 3,
                "object_threshold_median": 0.8,
                "fn_rate_max": 0.0,
                "fp_rate_mean": 0.2,
                "balanced_accuracy_mean": 0.9,
            },
            {
                "number": 2,
                "state": "PRUNED",
                "value_0": None,
                "value_1": None,
                "value_2": None,
            },
        ]
    )

    candidates = optuna_trials_to_candidate_configs(
        trials,
        n_per_matrix_family=0,
        n_overall=0,
        target_class="peanut",
        non_target_label="almond",
        selection_strategy="04B_optuna_search",
    )

    assert len(candidates) == 1
    row = candidates.iloc[0]
    assert row["matrix_family"] == "pixel_matrix"
    assert row["candidate_source"] == "04B_optuna_search"
    assert row["object_threshold"] == 0.8
    assert row["fn_rate"] == 0.0
    assert row["fp_rate"] == 0.2
    assert row["balanced_accuracy"] == 0.9
