# Bayesian Neural Network Workshop

Brief intro to BNNs using terminology of [Pyro](http://pyro.ai/examples/intro_long.html), showcasing [TyXe](https://github.com/TyXe-BDL/TyXe/).
The introductory notebook is a compilation of the practical [Pyro Tutorials](http://pyro.ai/examples/).


## Installation on Octane

```bash
# make sure you're a member of the hacking-nn group (you should be if youre reading this ...)

# go to remote (going via cegate is okay)
ssh octane

# clone recursively:
git clone -recursive https://cegit.ziti.uni-heidelberg.de/hacking-neural-networks/bnn/
cd bnn/

# ----- Optional: Create Virtual Environment -----
# Option 1: Conda. Use this if your Python is not version 3.8.
conda create --name bayes --python=3.8.10
conda activate bayes

# Option 2: Virtualenv. The folder "bayes" handily is in the gitignore
virtualenv bayes
source bayes/bin/activate

# ----- Install dependencies -----
bash install.sh # adds ~ 2.5 GB to empty environment

# ----- Install IPyKernel if using virtual environment -----
python3 -m ipykernel install --user --name bayes

# ----- Run jupyter -----
jupyter notebook --no-browser --port XXXX # choose your own port > 1024

# in another shell on remote, run:
jupyter notebook list # copy the displayed token

# locally (on cegate if connected through it), run:
ssh USERNAME@octane -NL 1234:localhost:XXXX

# if you ran the above command on cegate, run:
ssh USERNAME@cegate.ziti.uni-heidelberg.de -NL 1234:localhost:1234

# open jupyter in browser on your machine
http://localhost:1234/

# enter the copied token if prompted

# open bnn_workshop.ipynb

# select kernel > change kernel > bayes (or your environment name)
```

NOTE: The install itself requires ~2.5 GB if torch/numpy/pyro/etc arent present; and we will require another 500 MB for MNIST + CIFAR10.

