from pathlib import Path
from typing import List, Optional

#import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
#from matplotlib.patches import Rectangle
from soundevent import arrays, data

from batdetect2.data.compat import load_annotation_project
from batdetect2.data.labels import ClassMapper
from batdetect2.train.preprocess import (
    PreprocessingConfig,
    preprocess_annotations,
)
from class_mapper import Mapper



def gen_newFormat_annotations(data_dir):
    proj = load_annotation_project(
        data_dir
        / "ann",  # Specify the folder containing annotation files (json files)
        audio_dir= data_dir
        / "wav",  # Specify the folder containing audio recordings
    )
    
    # Verify the number of loaded clips and the data directory
    print(f"Loaded {len(proj.clip_annotations)} annotated clips from {data_dir}")
    
    config = PreprocessingConfig(
        target_samplerate=256_000,  # Target sample rate for resampling audio
        scale_audio=False,  # Whether to scale the raw audio values to -1,-1 range
        fft_win_length=512 / 256_000,  # Window length for FFT in seconds
        fft_overlap=0.75,  # Overlap between FFT windows
        max_freq=120_000,  # Maximum frequency to include in spectrogram
        min_freq=200,  # Minimum frequency to include in spectrogram
        spec_scale="pcen",  # Spectrogram scaling method ("pcen", "db", or None)
        denoise_spec_avg=True,  # Whether to apply spectral denoising
        max_scale_spec=False,  # Whether to max-scale the spectrogram
        duration=1.05,  # Duration (in seconds) of each training example
        spec_height=128,  # Height of the spectrogram (number of frequency bins)
        spec_time_period=0.001,  # Time resolution of the spectrogram (seconds per pixel)
    )


    # Store the configuration in a JSON file for reference
    config.to_file(data_dir / "preprocessed/config.json")

    preprocess_annotations(
        proj.clip_annotations,  # Provide a list of clip annotations
        data_dir / "preprocessed",  # Output directory
        class_mapper=Mapper(),  # The custom Mapper we defined earlier
        config=config,  # Preprocessing configuration (either default or custom)
        replace=True,  # Replace any existing preprocessed data
    )

if __name__ == '__main__':   
    # Data Directory Setup
    data_dir = Path(
        "F:/bat/trainingBd2"
    )  # Path to the example data directory
    gen_newFormat_annotations(data_dir)
