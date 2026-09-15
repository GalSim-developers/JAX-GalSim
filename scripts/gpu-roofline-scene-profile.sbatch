#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=8g
#SBATCH --mail-user=ismael@umd.edu
#SBATCH --mail-type=BEGIN,END

module purge
module load cuda cudnn
unset LD_LIBRARY_PATH
source /home/ismael/scratch/JAX-GalSim/.venv/bin/activate

OUT_DIR="./scripts/output_sep2026" 

uv run python3 scripts/profile_scene_gpu.py --stamp-slen-bins-str "63,83,103,250" --max-n-gals-bins-str "65,20,5,3" --fft-size-bins-str "128,128,128,256" --image-slen 250 --max-n-gals-global 150 --out-dir ${OUT_DIR} --max-n-iters 5 --seed 42 --scan-or-vmap 'scatter-vmap' --catsim-fpath "../OneDegSq.fits" --n-samples 100

uv run python3 scripts/profile_scene_gpu.py --stamp-slen-bins-str "63,83,103,250,1000" --max-n-gals-bins-str "1050,300,40,20,3" --fft-size-bins-str "128,128,128,256,1024" --image-slen 1000 --max-n-gals-global 2000 --out-dir ${OUT_DIR} --max-n-iters 5 --seed 42 --scan-or-vmap 'scatter-vmap' --catsim-fpath "../OneDegSq.fits" --n-samples 100 
