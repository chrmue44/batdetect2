set DATA_SET=GermanBats093
set AUDIO_PATH=F:\bat\trainingBd2\wav\
set ANN_PATH=F:\bat\trainingBd2\ann\
set OUT_PATH=F:\bat\trainingBd2\split\
set SPLIT=0.18
set SEED=123478
set CLASSES_IN="Myotis mystacinus;Myotis brandtii;Plecotus auritus;Plecotus austriacus;Pipistrellus kuhlii;Pipistrellus nathusii"
set CLASSES_OUT="Mbart;Mbart;Plecotus;Plecotus;Pipistrellus nathusii;Pipistrellus nathusii"
call _venv\Scripts\activate.bat
cd bat_detect\finetune
python prep_data_finetune.py %DATA_SET% %AUDIO_PATH% %ANN_PATH% %OUT_PATH% --percent_val %SPLIT% --rand_seed %SEED% --input_class_names %CLASSES_IN% --output_class_names %CLASSES_OUT%
cd ..\..
call _venv\Scripts\deactivate.bat
pause
