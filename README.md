<h1 align="center">
  Multi-Physics-Enhanced Bayesian Inverse Analysis:
  Information Gain from Additional Fields
</h1>

<div align="center">

[![Pipeline](https://github.com/leahaeusel/mpebia/actions/workflows/pipeline.yml/badge.svg)](https://github.com/leahaeusel/mpebia/actions/workflows/pipeline.yml)
[![Coverage badge](https://github.com/leahaeusel/mpebia/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/leahaeusel/mpebia/tree/python-coverage-comment-action-data)

</div>

This repository holds the code and scripts used to compute and plot any results from the publication "Multi-Physics-Enhanced Bayesian Inverse Analysis: Information Gain from Additional Fields".


## Installation

After installing miniforge, execute the following steps:

- Create a new conda environment based on the [`environment.yml`](./environment.yml) file:
```
mamba env create -f environment.yml
```

- Activate your newly created environment:
```
conda activate mpebia
```

- Install all mpebia requirements with:
```
pip install -e .
```

- *Optional:* Install pre-commit hooks:
```
pre-commit install
```

- Clone the QUEENS repository from https://github.com/queens-py/queens, navigate to its base directory, and check out the following commit:
```
git checkout 699d819baf0bb77ca75aaf2c376428b4ea132d2f
```

- Next, install QUEENS:
```
pip install .[safe,fourc]
```


### 4C

The 4C commit `a1f1d02ca1a0786720cadfcb731fffc5fe19b32b` from https://github.com/4C-multiphysics/4C was used for any evaluations of the porous medium model.


## How to cite

If you use this code or parts of it, please cite the following article:

```bib
@misc{haeusel2025,
      title={Multi-Physics-Enhanced Bayesian Inverse Analysis: Information Gain from Additional Fields},
      author={Lea J. Haeusel and Jonas Nitzler and Lea J. Köglmeier and Wolfgang A. Wall},
      year={2025},
      eprint={TODO},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/TODO},
}
```

## License

This project is based on PySkel by David Rudlstorfer.
The original project is licensed under the MIT License.
This project is also licensed under the MIT license.
For further information check [`LICENSE`](./LICENSE).
