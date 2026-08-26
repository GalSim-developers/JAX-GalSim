#!/bin/bash
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=8g
#SBATCH --mail-user=ismael@umd.edu
#SBATCH --mail-type=BEGIN,END

module purge
unset LD_LIBRARY_PATH
source /home/ismael/scratch/JAX-GalSim/.venv/bin/activate

# uv run python3 scripts/timing_tests_general.py --stamp-slen-bins-str '63,83,103,250' --max-n-gals-bins-str '65,20,5,3' --fft-size-bins-str "128,128,128,256" --n-samples 500 --image-slen 250 --max-n-gals-global 150 --outdir "./scripts/output_aug2026_outliers" --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --include-outliers --scan-or-vmap 'vmap' --cpu-or-gpu 'gpu' --catsim-fpath "../OneDegSq.fits"


# uv run python3 scripts/timing_tests_general.py --stamp-slen-bins-str '63,83,103,250,1000' --max-n-gals-bins-str '1050,300,40,20,3' --fft-size-bins-str "128,128,128,256,1024" --n-samples 500 --image-slen 1000 --max-n-gals-global 2000 --outdir "./scripts/output_aug2026_outliers" --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --include-outliers --scan-or-vmap 'vmap' --cpu-or-gpu 'gpu' --catsim-fpath "../OneDegSq.fits"

uv run python3 scripts/timing_tests_general.py --stamp-slen-bins-str '63,83,121' --max-n-gals-bins-str '65,20,10' --fft-size-bins-str "128,128,128" --image-slen 250 --max-n-gals-global 150 --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --scan-or-vmap 'vmap' --cpu-or-gpu 'gpu' --catsim-fpath "../OneDegSq.fits" --outdir "./scripts/output_aug2026_outliers" --n-samples 500

uv run python3 scripts/timing_tests_general.py --stamp-slen-bins-str '63,83,123' --max-n-gals-bins-str '1050,300,50' --fft-size-bins-str "128,128,128" --image-slen 1000 --max-n-gals-global 2000 --max-n-iters 5 --psf-type "vary-moffat" --check-stamp-sizes --seed 45 --scan-or-vmap 'vmap' --cpu-or-gpu 'gpu' --catsim-fpath "../OneDegSq.fits" --outdir "./scripts/output_aug2026_outliers" --n-samples 500



# --scan-or-vmap 'vmap' --cpu-or-gpu 'gpu' --catsim-fpath "../OneDegSq.fits" --outdir "./scripts/output_aug2026_outliers" --n-samples 500
