# Paremeterisation-users

A userfriendly lightweight implementation for the parameterisation models introduced in [PAPER].

## Installation:

```Shell
git clone [Repo name]
cd [Repo name]
pip install -e  . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Getting started

To get started check out the jupyter notebook in the example folder.
The basic syntax to get started is:

```Python
from ParaPro import NN_predictor

eta = 3.082805; Lp_LD = 100; delta = 1; Cp = 20
X_in = eta, Lp_LD, delta, Cp

model = NN_predictor()
model.predict([X_in])
```
