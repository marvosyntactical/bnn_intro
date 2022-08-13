
# Bayesian Neural Network Workshop

Brief intro to BNNs using terminology of [Pyro](http://pyro.ai/examples/intro_long.html), showcasing [TyXe](https://github.com/TyXe-BDL/TyXe/).
The introductory notebook is a compilation of the practical [Pyro Tutorials](http://pyro.ai/examples/).

## How to run the notebook on a remote cluster
And access it through your browser, using port forwarding:

```bash
# go to remote
ssh yourserver

# clone recursively:
git clone --recursive https://github.com/marvosyntactical/bnn
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

# Start a shell session on a gpu node on your Cluster, e.g. using SLURM:
srun --mem 16000 --gres=gpu:1 --time=0-02:30:00 -w NODENAME -p partition --pty bash      

# ----- Run jupyter there -----
jupyter notebook --no-browser --port 1234 

# copy one of the URLs displayed by the jupyter command, it should have this format:
http://localhost:1234/?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ----- Do Port Forwarding -----

# on yourserver, run
ssh USERNAME@NODENAME -NL 1234:localhost:1234

# ----- In Your Browser -----

# 1. open jupyter in browser on your machine
# (paste the copied link)

# 2. open bnn_workshop.ipynb

# 3. select kernel > change kernel > bayes (or your environment name)
```

NOTE: The install itself requires ~2.5 GB if torch/numpy/pyro/etc arent present; and we will require another 500 MB for MNIST + CIFAR10.


```
