"Q-Exponential Latent Variable Model"

import os
import numpy as np
import scipy.sparse.linalg as spsla
import matplotlib.pylab as plt

import torch
from tqdm.notebook import trange

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models.qeplvm.latent_variable import *
from gpytorch.models.qeplvm.bayesian_qeplvm import BayesianQEPLVM
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import QExponentialPrior
from gpytorch.likelihoods import QExponentialLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateQExponential

# POWER = torch.tensor(1.0)

# Setting manual seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

POWER = torch.tensor(1.0, device=device)

# load CT data
loaded=np.load(os.path.join('./','CT_obs_proj60.npz'),allow_pickle=True)
proj=loaded['proj'][0]
projector=torch.tensor(proj.todense(), dtype=torch.float32, device=device)
sinogram=torch.tensor(loaded['obs'], device=device).unsqueeze(-1)
nzvar=torch.tensor(loaded['nzvar']); truth=loaded['truth']


# define model

class bQEPLVM(BayesianQEPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, lsqr=False):
        self.power = POWER
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim, device=device)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape, power=self.power)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim, device=device)  # shape: N x Q
        prior_x = QExponentialPrior(X_prior_mean, torch.ones_like(X_prior_mean), power=self.power)

        # Initialise X c
        if lsqr == True:
             X_init = torch.nn.Parameter(torch.tensor(spsla.lsqr(A=proj, b=sinogram.cpu(), damp=1)[0], dtype=torch.float32, device=device)) # Initialise X to least square solution
        else:
             X_init = torch.nn.Parameter(torch.randn(latent_dim, device=device)) #torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # # Kernel (acting on latent dimensions)
        # self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
        # Kernel (acting on transformed latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=n)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n))

    def forward(self, X):
        proj_X = torch.matmul(X, projector.T)
        mean_ = self.mean_module(proj_X)
        covar_ = self.covar_module(proj_X)
        dist = MultivariateQExponential(mean_, covar_, power=self.power)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


N = len(sinogram)
data_dim = sinogram.shape[1]
latent_dim = projector.shape[1] # data_dim
n_inducing = 100
lsqr = True

# Model
model = bQEPLVM(N, data_dim, latent_dim, n_inducing, lsqr=lsqr)

# Likelihood
likelihood = QExponentialLikelihood(batch_shape=model.batch_shape, power=POWER)

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, model, num_data=N)

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.1)

# set device
# POWER = POWER.to(device)
projector = projector.to(device)
sinogram = sinogram.to(device)
model = model.to(device)
likelihood = likelihood.to(device)
mll = mll.to(device)

# Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
# using the optimizer provided.

loss_list = []
iterator = trange(1000, leave=True)
for i in iterator:
    optimizer.zero_grad()
    sample = model.sample_latent_variable()  # a full sample returns latent x across all N
    if sample.ndim==1: sample = sample.unsqueeze(0)
    output = model(sample)
    loss = -mll(output, sinogram.T).sum()
    iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
    loss.backward()
    optimizer.step()
    loss_list.append(loss.item())


# plot results

plt.figure(figsize=(20, 7))
plt.set_cmap('gray')#'Greys')

plt.subplot(131)
plt.imshow(truth, extent=[0, 1, 0, 1])
plt.title('Truth')

plt.subplot(132)
X = (model.X.q_mu if hasattr(model.X, 'q_mu') else model.X.X).detach().cpu().numpy().reshape(truth.shape,order='F')
plt.imshow(X, extent=[0, 1, 0, 1])
plt.title('Solution')


plt.subplot(133)
plt.plot(loss_list, label='batch_size=100')
plt.title('Neg. ELBO Loss', fontsize='small')
# plt.show()
plt.savefig('./results/CT_QEP-LVM.png',bbox_inches='tight')