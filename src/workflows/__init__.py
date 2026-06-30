from src.workflows.pixel_projection import (
    build_training_matrix,
    build_projection_pixel_matrix,
    fit_one_class_peanut_simca,
    predict_pixels_with_simca,
    fit_one_class_simca,
)

from src.workflows.simca_optuna import (
    make_simca_optuna_objective,
    run_optuna_simca_pixel_optimization,
    optuna_trials_dataframe,
    best_completed_trial_row,
    refit_optuna_best_trial,
    close_optuna_study,
)

from src.workflows.pca_comparison import (
    compare_pca_representations,
    add_pca_selection_score,
)

from src.workflows.pca_diagnostic import (
    class_separation_scores,
    compute_pca_summary_metrics,
)

from src.workflows.simca_pixel_grid import (
    make_target_train_filters,
    make_peanut_train_filters,
    run_single_simca_pixel_projection,
    run_simca_pixel_projection_grid,
    refit_best_grid_row,
    refit_selected_simca_row,
    refit_selected_simca_configs,
)

from src.workflows.simca_cv_calibration import (
    calibrate_simca_thresholds_cv,
    fit_final_simca_model,
    project_pixels_with_rule_variants,
    summarize_cv_calibration,
    run_simca_empirical_rule_grid,
    refit_empirical_cv_rule_row,
)