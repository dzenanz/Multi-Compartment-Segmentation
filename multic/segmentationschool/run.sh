#!/bin/sh
#SBATCH --account=banff-aid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --partition=community
#SBATCH --gpus=rtx6000:2
#SBATCH --time=72:00:00
#SBATCH --output=job%A.log
#SBATCH --job-name="BANFF-AID-Training1"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST
echo "SLURM_NNODES="$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR

echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited
which python

echo "Launch job"
CUDA_LAUNCH_BLOCKING=1
python3 segmentation_school.py --option train --base_dir /home/local/KHQ/dzenan.zukic/Histo/ --init_modelfile /home/local/KHQ/dzenan.zukic/Histo/model_0214999.pth --training_data_dir /data/Public/banff-aid/TrainBig/ --train_steps 200000 --eval_period 500 --num_workers 2

echo "SLURM script Done!"
