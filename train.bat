rem Name of the data set
set DATA_SET=GermanBats03
rem path to the folder containing the WAV files
set AUDIO_PATH=F:\bat\trainingBd2\wav\
rem file name of json file of training set
set TRAIN_DATA=F:\bat\trainingBd2\split\%DATA_SET%_TRAIN.json
rem file name of json file of test set
set TEST_DATA=F:\bat\trainingBd2\split\%DATA_SET%_TEST.json
rem path to store the result data file
set OUT_PATH=F:\bat\trainingBd2\model\
rem path to the pretrained model 
set TRAINED_MODEL=../../models/Net2DFast_UK_same.pth.tar
rem number of epochs
set EPOCHS=50
rem set flag to train from scratch
set O_SCRATCH=
rem set O_SCRATCH=----train_from_scratch
rem set flag to train only last layer
set O_LAST_L=
rem set O_LAST_L=--finetune_only_last_layer
call _venv\Scripts\activate.bat
cd bat_detect\finetune
echo python finetune_model.py %AUDIO_PATH% %TRAIN_DATA% %TEST_DATA% %TRAINED_MODEL% --op_model_name %DATA_SET%.tar --num_epochs %EPOCHS% %O_SCRATCH% %O_LAST_L%
python finetune_model.py %AUDIO_PATH% %TRAIN_DATA% %TEST_DATA% %TRAINED_MODEL% --op_model_name %DATA_SET%.tar --num_epochs %EPOCHS% %O_SCRATCH% %O_LAST_L%
cd ..\..
call _venv\Scripts\deactivate.bat
