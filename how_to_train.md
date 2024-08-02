# How to train the model BatDetect2

The documents provides a very brief description of all steps needed to train the model BatDetect2 with own data.  
It was tested on a Windows 10 system.

## Prepare Test Data

### Installation of GUI (only once)

- download the repository https://github.com/macaodha/batdetect2_GUI
- create virtual environment 

		python -m venv _venv
- activate the virtual environment

		_venv\Scripts\Activate

- install all the needed libraries (using the requirements.txt file in the GUI repo)

		pip install -r requirements.txt

### Annotation

- limit the length for all training data files to (e.g.) 1 sec (the performance and memory consumption depends on the length to the power of 2, so don't change the value too liberally)
	  
- edit the list of classes to annotate in the file batdetect2_gui/config.py: CLASS_NAMES=...
- start the webserver with the command 

        call _venv\Scripts\activate
        python application.py

- start browser and goto page localhost:8000
- annotate clips with tool BatDetect_GUI:
- copy all annotated wav files to a single folder  <somepath>/wav
- copy all annotation json files to another single folder <somepath>/wav

### Convert Annotation Data to "New" Format

- download branch 'train' from repoistory https://github.com/macaodha/batdetect2
- setup config in in file convert_annotations_old_to_new.py (around line 31)<br>
  be aware that:
	- min_freq must be lower than the lowest annotated frequency used during annotation (lower margin of bounding box)
	- duration must be slightly longer than selected clip length (in this case 1s --> duration = 1.05)

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
	
- setup the root path name for wav files and annotation files in file convert_annotations_old_to_new.py (around line 60)

		data_dir = Path(
			"F:/bat/trainingBd2"
		)  # Path to the example data directory

- adapt the classes that the model shall detect in the file class_mapper.py. <br> In the given example all "echolocation" events are mapped to specfic species, all "social" events are mapped to a single class "social". All "Feeding" events are mapped to a single class "feeding". With more and more data one could try to map "social" events to individual species as well
		
- run script convert_annotations_old_to_new.py
	
	python convert_annotations_old_to_new.py
	
- after finishing the script the folder <selectedRootPath>/"preprocessed" should be created and contain data files (*.nc)


## Training the model

### Installation (only once)

- create a virtual environment for the model (may run in the same as the GUI, but I didn't test)
- install all the needed libraries (using the requirements.txt file in the BatDetect2 repo)

		pip install -r requirements.txt

### Training

- run training.py 
- to inspect the improvement of the performance of the model open another terminal and start

		tensorboard  --logdir=<rootPath>/preprocessed/lightning_logs
		
- open web browser and got to page localhost:6006

 