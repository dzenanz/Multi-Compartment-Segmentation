set USER=Dzenan
set PROJECT=multic_segment
set BASEDIR=M:/Histo/Work
set DATADIR=M:/Histo/TrainSmall
set MODELDIR=M:/Histo/Work
set CONTAINER=dzenanz/compreps:hpg_training
set CUDA_LAUNCH_BLOCKING=1
docker run -a STDIN -a STDOUT -a STDERR --gpus=all -v M:/Histo/Multi-Compartment-Segmentation/:/exec/ -v %DATADIR%/:/data -v %MODELDIR%/:/model/ --entrypoint python3 %CONTAINER% /exec/multic/segmentationschool/segmentation_school.py --option train --base_dir /model --init_modelfile /model/model_0214999.pth --training_data_dir /data --train_steps 1000 --eval_period 250 --num_workers 0
GOTO :EOF

REM Interactive debugging:
docker run -it --gpus=all -v M:/Histo/Multi-Compartment-Segmentation/:/exec/ -v %DATADIR%/:/data -v %MODELDIR%/:/model/ --entrypoint /bin/bash %CONTAINER%
python3 /exec/multic/segmentationschool/segmentation_school.py --option train --base_dir /model --init_modelfile /model/model_0214999.pth --training_data_dir /data --train_steps 1000 --eval_period 250 --num_workers 0
