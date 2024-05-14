"Deep Q-Exponential Process Model"

import os
import numpy as np
import scipy.sparse.linalg as spsla
import math
from matplotlib import pyplot as plt

import torch
from tqdm.notebook import trange

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.qeplvm.latent_variable import *
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.priors import QExponentialPrior
from gpytorch.likelihoods import MultitaskQExponentialLikelihood

# Setting manual seed for reproducibility
torch.manual_seed(2024)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

POWER = torch.tensor(1.0, device=device)

# load CT data
loaded=np.load(os.path.join('./','CT_obs_proj60.npz'),allow_pickle=True)
proj=loaded['proj'][0]
projector=torch.tensor(proj.todense(), dtype=torch.float32, device=device)
sinogram=torch.tensor(loaded['obs'], device=device).unsqueeze(-1)
nzvar=torch.tensor(loaded['nzvar']); truth=loaded['truth']


# Here's a simple standard layer
class DQEPHiddenLayer(DeepQEPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=100, linear_mean=True, **kwargs):
        self.power = POWER
        inducing_points = torch.randn(output_dims, num_inducing, input_dims, device=device)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
            power=self.power
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        n = kwargs.pop('n',1); 
        self.mean_module = ConstantMean() if not linear_mean else LinearMean(n)#input_dims)
        # self.covar_module = ScaleKernel(
        #     RBFKernel(
        #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
        #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
        #         ), batch_shape=batch_shape, ard_num_dims=input_dims
        #     )
        # )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=None),#input_dims),
            batch_shape=batch_shape, ard_num_dims=None,
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            )
        )
        
        # LatentVariable (c)
        data_dim = output_dims; latent_dim = input_dims
        latent_prior_mean = torch.zeros(n, latent_dim, device=device)
        latent_prior = QExponentialPrior(latent_prior_mean, torch.ones_like(latent_prior_mean), power=self.power)
        latent_init = torch.nn.Parameter(kwargs.pop('latent_init', torch.randn(latent_dim, device=device)))
        self.latent_variable = VariationalLatentVariable(n, data_dim, latent_dim, latent_init, latent_prior)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, latent_init)
        # X = MAPLatentVariable(n, latent_dim, latent_init, latent_prior)

    def forward(self, x, projection=None):
        x_ = x if projection is None else torch.matmul(x, projection.T)
        mean_x = self.mean_module(x_)
        covar_x = self.covar_module(x_)
        return MultivariateQExponential(mean_x, covar_x, power=self.power)

# define the main model
num_tasks = sinogram.size(-1)
num_hidden_dqep_dims = 2


class MultitaskDeepQEP(DeepQEP):
    def __init__(self, n, latent_dim):
        hidden_layer = DQEPHiddenLayer(
            input_dims=latent_dim,
            output_dims=num_hidden_dqep_dims,
            linear_mean=False,
            n=n,
            latent_init=torch.tensor(spsla.lsqr(A=proj, b=sinogram.cpu(), damp=1.0)[0], dtype=torch.float32, device=device)
        )
        last_layer = DQEPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False,
            n=n
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=num_tasks, power=POWER)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs, projection=projector)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)

n = sinogram.shape[0]
latent_dim = projector.shape[1]
model = MultitaskDeepQEP(n, latent_dim)

# training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=n))

# set device
# POWER = POWER.to(device)
# projector = projector.to(device)
# sinogram = sinogram.to(device)
model = model.to(device)
mll = mll.to(device)

loss_list = []
num_epochs = 500
iterator = trange(num_epochs, leave=True)
for i in iterator:
    optimizer.zero_grad()
    sample = model.hidden_layer.latent_variable()
    if sample.ndim==1: sample = sample.unsqueeze(0)
    output = model(sample)
    loss = -mll(output, sinogram.T).sum()
    iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
    loss.backward()
    optimizer.step()
    # record the loss and the best model
    loss_list.append(loss.item())
    if i==0:
        min_loss = loss_list[-1]
        optim_model = model.state_dict()
    else:
        if loss_list[-1] < min_loss:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()

# load the best model
model.load_state_dict(optim_model)
model.eval()
# plot results

plt.figure(figsize=(20, 7))
plt.set_cmap('gray')#'Greys')

plt.subplot(131)
plt.imshow(truth, extent=[0, 1, 0, 1])
plt.title('Truth')

plt.subplot(132)
X = getattr(model.hidden_layer.latent_variable, 'q_mu' if isinstance(model.hidden_layer.latent_variable, VariationalLatentVariable) else 'X')
X = X.detach().cpu().numpy().reshape(truth.shape,order='F')
plt.imshow(X, extent=[0, 1, 0, 1])
plt.title('Solution')


plt.subplot(133)
plt.plot(loss_list, label='batch_size=100')
plt.title('Neg. ELBO Loss', fontsize='small')
# plt.show()
plt.savefig('./results/CT_Deep_QEP.png',bbox_inches='tight')