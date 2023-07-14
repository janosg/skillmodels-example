# skillmodels-example


Self-contained repo to reproduce the getting started tutorial with skillmodels.


## Installation via environment

1. Install anaconda
2. Open a terminal in the root directory of this repo and run
```bash
conda env create -f enviroment.yml
conda activate skillmodels
```
3. Open tutorial.ipynb in a browser or your editor of choice and select the correct
environment as Kernel.


## Alternative installation

Alternatively, you can install all programs from the environment file into your conda
root environment.

Make sure to install skillmodels via:

```bash
pip install git+https://github.com/janosg/skillmodels.git@main
```

The latest version that was released via conda is quite outdated and should not be
used.