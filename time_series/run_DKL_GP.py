"Deep Kernel Learning Gaussian Process Model"

import os
import math
from matplotlib import pyplot as plt

import torch
import tqdm
from torch.nn import Linear

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# Setting manual seed for reproducibility
torch.manual_seed(2024)

# generate data
train_x = torch.linspace(0, 1, 100)

f = {'step': lambda ts: torch.tensor([1*(t>=0 and t<=1) + 0.5*(t>1 and t<=1.5) + 2*(t>1.5 and t<=2) for t in ts]),
     'turning': lambda ts: torch.tensor([1.5*t*(t>=0 and t<=1) + (3.5-2*t)*(t>1 and t<=1.5) + (3*t-4)*(t>1.5 and t<=2) for t in ts])}

train_y = torch.stack([
    f['step'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
    f['turning'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
], -1)

train_x = train_x.unsqueeze(-1)


# define the NN feature extractor
data_dim = train_x.size(-1)
num_features = train_y.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, num_features))

feature_extractor = LargeFeatureExtractor()


# define the GP layer
class GaussianProcessLayer(ApproximateGP):
    def __init__(self, input_dims, output_dims, grid_bounds=(-10., 10.), grid_size=128, linear_mean=True):
        batch_shape = torch.Size([output_dims])
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=batch_shape
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = IndependentMultitaskVariationalStrategy(
            GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=output_dims,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        # self.covar_module = ScaleKernel(
        #     RBFKernel(
        #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
        #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
        #         ), batch_shape=batch_shape, ard_num_dims=input_dims
        #     )
        # )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None,
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            )
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


# define the main model
num_tasks = train_y.size(-1)

class MultitaskDKLGP(gpytorch.Module):
    def __init__(self, feature_extractor, output_dims, grid_bounds=(-10., 10.)):
        super(MultitaskDKLGP, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(input_dims=num_features, output_dims=output_dims, grid_bounds=grid_bounds)

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = likelihood(model(test_x)).to_data_independent_dist()

        # return preds.mean.mean(0), preds.variance.mean(0)
        return preds.mean, preds.variance

likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
model = MultitaskDKLGP(feature_extractor, output_dims=num_tasks)


# training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = VariationalELBO(likelihood, model.gp_layer, num_data=train_y.size(0))

loss_list = []
num_epochs = 1000
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
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
# Make predictions
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    mean, var = model.predict(test_x)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

# Plot results
fig, axs = plt.subplots(1, num_tasks+1, figsize=(4 * (num_tasks+1), 4))
for task, ax in enumerate(axs):
    if task < num_tasks:
        ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
        ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
        ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.set_ylim([-1, 3])
        ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'])
        ax.set_title(f'Task {task + 1}: '+list(f.keys())[task]+' function')
    else:
        ax.plot(loss_list)
        ax.set_title('Neg. ELBO Loss')
fig.tight_layout()

# plt.show()
plt.savefig('./results/ts_DKLGP.png',bbox_inches='tight')