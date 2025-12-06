#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=4G      # Required memory per GPU (specify how many GB)
#SBATCH --time=4:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J hmc                # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32

# Execute the command

##Old lr: 0.0002
module load anaconda
conda activate rl-sparse

# cd /home/venkat97/cs593-cvd/ArtStyleToArtStyle
cd /home/tyalaman/ArtStyleToArtStyle
python train_baseline.py  --root datasets \
  --dataset_name cezanne2photo \
  --model_type improved \
  --batch_size 1 \
  --num_workers 4 \
  --load_size 150 \
  --crop_size 128 \
  --lr 0.0002 \
  --beta1 0.5 \
  --beta2 0.999 \
  --num_epochs 80 \
  --n_epochs_decay 40\
  --lambda_cycle 10.0 \
  --lambda_identity 0.5 \
  --lambda_content 1.0 \
  --lambda_style 1.0 \
  --lambda_fm 1.0 \
  --checkpoint_dir checkpoints/cezanne2photo_improved\
  --sample_dir samples/cezanne2photo_improved\
  --seed 42