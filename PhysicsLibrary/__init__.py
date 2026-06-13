"""
PhysicsLibrary
--------------
Data processing and analysis library for Physics Analysis GUI.
"""

from .file_parser import (
    choose_file,
    detect_format,
    detect_format_file,
    load_dataset,
    load_dataset_file,
    DataFormat,
    Dataset,
)

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
    compute_slope_segment,
    compute_fft_slice,
    annotate_fft_peaks
)

from .models import (
    double_exponential_model,
    visibility_model,
)