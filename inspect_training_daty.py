from pathlib import Path
import xarray as xr

# Gather All Processed NetCDF Files
data_dir = Path("F:/bat/trainingBd2")
output_files = list((data_dir / "preprocessed").glob("*.nc"))
print(f"Found {len(output_files)} processed NetCDF files.")

# Load an Example NetCDF File
for x in range(0,10):
    example = xr.open_dataset(output_files[x])
    spectrogram = example["spectrogram"]  # The preprocessed spectrogram data
    print("spectrogram[{}] shape {}".format(x, spectrogram.shape))
    
    detection_array = example["detection"]
    print("detection[{}] shape {}".format(x, detection_array.shape))

    class_array = example["class"]  # Array indicating the class of the detection
    print("class[{}] shape {}".format(x, class_array.shape))
    
    size_array = example["size"]  # Array with the size of each detection
    print("size_array[{}] shape {}".format(x, size_array.shape))
    
    print ("-"*20)
