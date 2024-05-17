"Showcase Deep Gaussian Process Classification Model"

import os
import math
import numpy as np
from matplotlib import pyplot as plt

import torch
import tqdm
from torch.nn import Linear

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskDirichletClassificationLikelihood


# generate data
def gen_data(num_data, seed = 2019):
    torch.random.manual_seed(seed)

    x = torch.randn(num_data,1)
    y = torch.randn(num_data,1)

    u = torch.rand(1)
    # data_fn = lambda x, y: 1 * torch.sin(0.15 * u * 3.1415 * (x + y)) + 1
    # data_fn = lambda x, y: 1 * torch.cos(0.4 * u * np.pi * np.sqrt(x**2 + y**2)) + 1
    data_fn = lambda x, y: 1 * torch.cos(0.4 * u * np.pi * np.abs(x) + np.abs(y)) + 1
    latent_fn = data_fn(x, y)
    z = torch.round(latent_fn).long().squeeze()
    return torch.cat((x,y),dim=1), z, data_fn
n_train = 500
train_x, train_y, genfn = gen_data(n_train)
# plt.scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), c = train_y)
# plt.show()

# testing data
n_test = 50
test_d1 = np.linspace(-3, 3, n_test)
test_d2 = np.linspace(-3, 3, n_test)

test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)
test_labels = torch.round(genfn(test_x_mat, test_y_mat))
test_y = test_labels.view(-1)

# # show true boundary
# plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
# plt.show()

# Here's a simple standard layer
class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# define the main model
num_hidden_dgp_dims = 2

class DirichletDeepGP(DeepGP):
    def __init__(self, train_x, train_y, likelhood, num_classes):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x.shape[-1],
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_classes,
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        
        self.likelihood = likelhood

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
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


# we let the DirichletClassificationLikelihood compute the targets for us
likelihood = MultitaskDirichletClassificationLikelihood(train_y, learn_additional_noise=True)
model = DirichletDeepGP(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 500

# Find optimal model hyperparameters
model.train()
# likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
# mll = DeepLikelihood(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, model.likelihood.transformed_targets.T).sum()
    loss.backward()
    if i % 5 == 0:
        print('Iter %d/%d - Loss: %.3f  hiddenlayer lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.hidden_layer.covar_module.base_kernel.lengthscale.mean().item(),
            model.likelihood.second_noise_covar.noise.mean().item()
        ))
    optimizer.step()

# prediction
model.eval()
# likelihood.eval()

with gpytorch.settings.fast_pred_var(), torch.no_grad():
    test_dist = model(test_x)

    pred_means = test_dist.mean.mean(0)

# logits
fig, ax = plt.subplots(1, 3, figsize = (15, 5))

for i in range(3):
    im = ax[i].contourf(
        test_x_mat.numpy(), test_y_mat.numpy(), pred_means[:,i].numpy().reshape((n_test,n_test))
    )
    fig.colorbar(im, ax=ax[i])
    ax[i].set_title("Logits: Class " + str(i), fontsize = 20)
plt.savefig('./demo_DGP_cls_logits.png',bbox_inches='tight')

# # probabilities
# pred_samples = test_dist.sample(torch.Size((256,))).exp()
# probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean((0,1))
#
# fig, ax = plt.subplots(1, 3, figsize = (15, 5))
#
# levels = np.linspace(0, 1.05, 20)
# for i in range(3):
#     im = ax[i].contourf(
#         test_x_mat.numpy(), test_y_mat.numpy(), probabilities[:,i].numpy().reshape((n_test,n_test)), levels=levels
#     )
#     fig.colorbar(im, ax=ax[i])
#     ax[i].set_title("Probabilities: Class " + str(i), fontsize = 20)
# plt.savefig('./demo_DGP_cls_probabilities.png',bbox_inches='tight')

# boundaries
fig, ax = plt.subplots(1,2, figsize=(10, 5))

ax[0].contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
ax[0].scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), c = train_y)
ax[0].set_title('True Response', fontsize=20)

ax[1].contourf(test_x_mat.numpy(), test_y_mat.numpy(), pred_means.max(1)[1].reshape((n_test,n_test)))
ax[1].set_title('Estimated Response', fontsize=20)
plt.savefig('./demo_DGP_cls_boundaries.png',bbox_inches='tight')