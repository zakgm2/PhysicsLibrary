"""
PhysicsLibrary
--------------
Data processing and analysis library for Physics Analysis GUI.
"""

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("ZaksPhysicsLibrary")
except PackageNotFoundError:
    __version__ = "unknown"

from .file_parser_generic import load_any_file

from .dataset import (
    detect_format,
    detect_format_file,
    DataFormat,
    Dataset,
)

from .file_parser import (
    load_dataset,
    load_dataset_file,
)

from .loaders.pt2_loader import load_pt2

from .processing_TDT import (
    process_tdt_folder,
    validate_tdt_folder,
    get_tdt_struct,
    get_plot_data,
    correct_bleaching,
    denoise_signal,
    get_event_markers,
)

from .analysis import (
    get_zscore_slice,
    smooth_signal,
    bin_for_heatmap,
    compute_fft_slice,
    annotate_fft_peaks,
    compute_slope_segment,
    fit_model_to_segment,
    compute_event_zscore_peth,
    find_significant_peaks,
    find_peak_near_events,
)

from .splice import (
    splice_keep_inside,
    splice_cut_out,
)

from .models import (
    double_exponential_model,
    visibility_model,
    linear_model,
    single_exponential_model,
    exponential_rise_model,
    gaussian_model,
    sinusoidal_model,
)

from .text_field_study import (
    run_field_study_pipeline,
    load_field_study_folder,
    peek_fields,
    flag_low_quality,
    embed_text_fields,
    compute_delta_vector,
    compute_paired_similarity,
    permutation_test_similarity,
    wordcount_confound_check,
)

from .field_study_validation import (
    run_validation_pipeline,
    build_validation_summary,
    cohens_d,
    benjamini_hochberg,
    wordcount_controlled_regression,
    bootstrap_mean_ci,
    leave_one_out_sensitivity,
)