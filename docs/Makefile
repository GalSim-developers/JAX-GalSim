# Minimal Sphinx Makefile — uses uv run so all package deps are always available.

SPHINXBUILD  = uv run sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

.PHONY: help html clean livehtml

help:
	@echo "Usage:"
	@echo "  make html      Build the HTML documentation"
	@echo "  make clean     Remove the build directory"
	@echo "  make livehtml  Auto-rebuild on file changes (requires sphinx-autobuild)"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)

livehtml:
	uv run sphinx-autobuild $(SOURCEDIR) $(BUILDDIR)/html \
		--ignore "**/_build/**" \
		--ignore "**/.DS_Store"
