# Make all figures for the JAX-GalSim paper.
#
#   make -C scripts/paper            # all figures
#   make -C scripts/paper figure1    # one figure
#   make -C scripts/paper FIG_DIR=/some/other/folder
#
# Figures are only remade when their script (or a script they use) changes;
# `make -B` forces a full rebuild.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(HERE)/../..)

FIG_DIR ?= $(HOME)/Sync/manuscripts/jax-galsim-paper/figures
CATSIM ?= $(REPO)/output/OneDegSq.fits
CACHE_DIR ?= $(REPO)/output/paper
PYTHON ?= uv run --project $(REPO) python

FIG1 := $(FIG_DIR)/figure1_fixed_sizes.png
FIG2 := $(FIG_DIR)/figure2_full_scene.png

.PHONY: all figure1 figure2 clean

all: figure1 figure2

figure1: $(FIG1)
figure2: $(FIG2)

$(FIG1): $(HERE)/figure1_fixed_sizes.py $(HERE)/paper_style.py
	cd $(HERE) && $(PYTHON) figure1_fixed_sizes.py --out $@

$(FIG2): $(HERE)/figure2_full_scene.py $(HERE)/paper_style.py $(REPO)/scripts/draw_scene_functions.py
	cd $(HERE) && $(PYTHON) figure2_full_scene.py --out $@ --catsim-fpath $(CATSIM) --cache-dir $(CACHE_DIR)

clean:
	rm -f $(FIG1) $(FIG2)
