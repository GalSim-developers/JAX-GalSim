#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=8g
#SBATCH --mail-user=ismael@umd.edu
#SBATCH --mail-type=BEGIN,END

source /home/ismael/scratch/JAX-GalSim/.venv/bin/activate

module purge
module load cuda cudnn
unset LD_LIBRARY_PATH


python3 scripts/test_roofline_xprof.py --seed 42 --image-slen 250 --max-n-gals-global 150 --stamp-slen 61 --fft-size 128 

python3 scripts/test_roofline_xprof.py --seed 52 --image-slen 1000 --max-n-gals-global 2000 --stamp-slen 61 --fft-size 128
