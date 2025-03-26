set DATADIR=M:/Histo/TrainSmall
set MODELDIR=M:/Histo/Work
set CONTAINER=dzenanz/compreps:hpg_training
set CUDA_LAUNCH_BLOCKING=1
docker run -a STDIN -a STDOUT -a STDERR --gpus=all -v M:/Histo/Multi-Compartment-Segmentation/:/exec/ -v %DATADIR%/:/data -v %MODELDIR%/:/model/ --entrypoint python3 %CONTAINER% /exec/multic/segmentationschool/segmentation_school.py --option train --base_dir /model --init_modelfile /model/model_0214999.pth --training_data_dir /data --train_steps 1000 --eval_period 250 --num_workers 0

