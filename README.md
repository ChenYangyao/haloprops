# HaloProps: Halo Structural Property Calculator

Reference: Chen Y. et al. 2020 ([arXiv:2003.05137](https://arxiv.org/abs/2003.05137)).

## Install

### With PyPI

Use `pip` in your shell:
```bash
$ pip install haloprops
```
This installs `haloprop` as well as all its dependencies.

### Using this github repository

First download this repository. Then enter the root directory of this package, and use pip to install it
```bash
$ cd haloprops
$ pip install .
```

Alternatively, directly use setuptools
```bash
$ cd haloprops
$ python setup.py install
```

## Usage

A simple example of calculating halo concentration from ( the first PC of assembly history, halo mass, tidal anisotropy parameter, halo bias )

```python
from haloprops.structure import StructurePredictor

# fit template data
sp = StructurePredictor()
sp.fit('concentration')

# make prediction for 3 halos from their four properties ( the first PC of 
# assembly history, halo mass, tidal anisotropy parameter, halo bias )
X = [[ 1.55278125e+00,  2.78270723e+04,  1.88321865e-01, 4.02548960e+00],
  [ 3.08029618e-01,  2.88465363e+02,  1.81991570e-01, -1.14664588e-01],
  [ 3.46576614e-01,  1.57965881e+03,  4.11933281e-01, 2.77537420e+00]]
concentration = sp.predict(X).val
print(concentration)
```
Outputs is like
```txt
[8.62213406 8.98157622 7.72968138]
```

For the detail usage please refer to the doc string of class StructurePredictor.