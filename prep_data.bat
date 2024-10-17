set DATA_SET=GermanBats11
set AUDIO_PATH=F:\bat\trainingBd2\wav\
set ANN_PATH=F:\bat\trainingBd2\ann\
set OUT_PATH=F:\bat\trainingBd2\split\
set SPLIT=0.20
set SEED=123478
set CLASSES="Bat;Not Bat;Unknown;Barbastella barbastellus;Eptesicus nilssonii;Eptesicus serotinus;Hypsugo savii;Myotis alcathoe;Myotis daubentonii;Myotis dasycneme;Myotis emarginatus;Myotis myotis;Myotis mystacinus;Myotis natteri;Nyctalus leisleri;Nyctalus noctula;Pipistrellus kuhlii;Pipistrellus nathusii;Pipistrellus pipistrellus;Pipistrellus pygmaeus;Vespertilio murinus"
call _venv\Scripts\activate.bat
cd bat_detect\finetune
echo python prep_data_finetune.py %DATA_SET% %AUDIO_PATH% %ANN_PATH% %OUT_PATH% --percent_val %SPLIT% --rand_seed %SEED% --input_class_names %CLASSES% --output_class_names %CLASSES%
python prep_data_finetune.py %DATA_SET% %AUDIO_PATH% %ANN_PATH% %OUT_PATH% --percent_val %SPLIT% --rand_seed %SEED% --input_class_names %CLASSES% --output_class_names %CLASSES%
cd ..\..
call _venv\Scripts\deactivate.bat
pause
