# Bayesian Neural Network Workshop

Brief intro to BNNs using terminology of [Pyro](http://pyro.ai/examples/intro_long.html), showcasing [TyXe](https://github.com/TyXe-BDL/TyXe/).
The introductory notebook is a compilation of the practical [Pyro Tutorials](http://pyro.ai/examples/).


## Installation

```bash
# make sure you're a member of the group and to clone recursively:
git clone -recursive https://cegit.ziti.uni-heidelberg.de/hacking-neural-networks/bnn/
cd bnn/

# optionally create a virtualenv; the one below handily is in the gitignore
virtualenv bayes
source bayes/bin/activate

# ----- Install dependencies -----
bash install.sh # adds ~ 2.5 GB to empty venv
```

NOTE: The install itself requires ~2.5 GB if torch/numpy/pyro/etc arent present; and we will require another 500 MB for MNIST + CIFAR10.

