# Manages building, testing, and cleaning the code as well as running the code
# to generate the results and figures for the paper.

# CONFIGURATION
###############################################################################

# Set the package name
PACKAGE = fpinv

# Execute a notebook on the command line. Modifies the notebook inplace so the
# final execution order will be linear (from top to bottom). Running notebooks
# like this helps ensure that the final results are not a product of a
# non-linear execution order that can't be reproduced.
RUN = bash run_notebook.sh


# TARGETS
###############################################################################

help:
	@echo "Commands:"
	@echo ""
	@echo "  all        runs 'build'"
	@echo "  build      build and install the package"
	@echo "  clean      clean up the package build"
	@echo "  results    execute all of the results"
	@echo ""

all: build results

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache

build:
	unset PYTHONPATH # to avoid conflicts with local pygimli installations
	python setup.py install

results:
	make -C scripts/synthetic_case all
