# Manages building, testing, and cleaning the code as well as running the code
# to generate the results and figures for the paper.

# CONFIGURATION
###############################################################################

# Set the package name
PACKAGE = fpinv

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
	make -C scripts/field_case all
	make -C scripts/synthetic_case all
