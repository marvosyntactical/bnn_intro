# Bayesian Neural Network Workshop

Brief intro to BNNs using terminology of [Pyro](http://pyro.ai/examples/intro_long.html), showcasing [TyXe](https://github.com/TyXe-BDL/TyXe/).
The introductory notebook is a compilation of the practical [Pyro Tutorials](http://pyro.ai/examples/).


## Installation on Octane

```bash
# make sure you're a member of the hacking-nn group (you should be if youre reading this ...)

# go to remote (going via cegate is okay)
ssh octane

# clone recursively:
git clone --recursive https://cegit.ziti.uni-heidelberg.de/hacking-neural-networks/bnn.git
cd bnn/

# ----- Optional: Create Virtual Environment -----
# Option 1: Conda. Use this if your Python is not version 3.8.
conda create --name bayes python=3.8.10
conda activate bayes

# Option 2: Virtualenv. The folder "bayes" handily is in the gitignore
virtualenv bayes
source bayes/bin/activate

# ----- Install dependencies -----
bash install.sh # adds ~ 2.5 GB to empty environment

# ----- Install IPyKernel if using conda or virtualenv -----
python3 -m ipykernel install --user --name bayes

# Connect to a free gpu node
srun --mem 16000 --gres=gpu:1 --time=0-02:30:00 -w NODENAME -p dev --pty bash      

# ----- Run jupyter -----
jupyter notebook --no-browser --port 1234 

# copy one of the URLs displayed by the jupyter command, it should have this format:
http://localhost:1234/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ----- Do Port Forwarding Multiple Times -----

# on ceg-octane, run
ssh USERNAME@NODENAME -NL 1234:localhost:1234

# no output, blinking cursor is desired result

# locally (on cegate if connected through it), run:
ssh USERNAME@octane -NL 1234:localhost:1234

# if you ran the above command on cegate, run on your machine:
ssh USERNAME@cegate.ziti.uni-heidelberg.de -NL 1234:localhost:1234

# ----- In Your Browser -----

# 1. open jupyter in browser on your machine
# (paste the copied link)

# 2. open bnn_workshop.ipynb

# 3. select kernel > change kernel > bayes (or your environment name)
```

NOTE: The install itself requires ~2.5 GB if torch/numpy/pyro/etc arent present; and we will require another 500 MB for MNIST + CIFAR10.

