# Source code for producing the results and figures

The code is divided between Python modules in `fpinv` and Python scripts as well
as Jupyter notebooks in `scripts`. The modules implement the methodology and
code that is reused in different applications. The scripts perform the data
analysis and processing and generate the figures for the paper.

The `Makefile` automates all processes related to executing code.
Run the following to perform all actions from building the software to
generating the final figures:

    make all

## Building

Use the `Makefile` to build and install the software.

* Build and install:

        make build

## Generating results and figures

* Generate all results and figures:

        make results
