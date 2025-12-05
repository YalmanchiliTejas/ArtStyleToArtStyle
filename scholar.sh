#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=2G      # Required memory per GPU (specify how many GB)
#SBATCH --time=1:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J hmc                # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file
#SBATCH --account=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=01:00:00

# Execute the command
module load anaconda
conda activate CS587
cd /home/venkat97/cs593-cvd/ArtStyleToArtStyle
python train_baseline.py  --root datasets \
  --dataset_name cezanne2photo \
  --model_type baseline \
  --batch_size 1 \
  --num_workers 4 \
  --load_size 286 \
  --crop_size 256 \
  --lr 0.0002 \
  --beta1 0.5 \
  --beta2 0.999 \
  --num_epochs 1500 \
  --lambda_cycle 10.0 \
  --lambda_identity 5.0 \
  --lambda_content 1.0 \
  --lambda_style 1.0 \
  --lambda_fm 10.0 \
  --checkpoint_dir checkpoints/cezanne2photo_baseline \
  --sample_dir samples/cezanne2photo_baseline \
  --seed 42