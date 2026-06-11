"""
PhysicsLibrary
--------------
Top-level library for data analysis in physics.
"""
 
from .file_parser import (
    choose_file,
    detect_format,
    load_dataset,
    DataFormat,
    Dataset,
)
 
from .processing_TDT import (
    process_tdt_folder,
    validate_tdt_folder,
    get_tdt_struct,
    get_plot_data,
    correct_bleaching,
    get_event_markers,
    get_zscore_slice,
    smooth_signal,
    bin_for_heatmap,
    denoise_signal,
)
 
from .models import (
    double_exponential_model,
    visibility_model,
)