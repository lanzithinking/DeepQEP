"Gaussian Process Classification Model"

import os
import math
import torch
import numpy as np
from matplotlib import pyplot as plt

# gpytorch imports
import sys
from matplotlib.lines import fillStyles
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

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

# testing data
n_test = 50
test_d1 = np.linspace(-3, 3, n_test)
test_d2 = np.linspace(-3, 3, n_test)

test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)
test_labels = torch.round(genfn(test_x_mat, test_y_mat))
test_y = test_labels.view(-1)

# plot data with true boundary
sys.path.append('../')
from util.gpmkr_scatter import gpmkr_scatter
os.makedirs('./results', exist_ok=True)
mkrs = ['^', 'o', 'v']
fig = plt.figure(figsize=(5, 5))
plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
gpmkr_scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), m = [mkrs[int(i)] for i in train_y], facecolors='none', edgecolors='r')
plt.title('True Labels', fontsize=18)
plt.savefig('./results/cls_truth.png',bbox_inches='tight')
# plt.show()

# Define model
# We will use the simplest form of GP model, exact inference
class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((num_classes,))),
            batch_shape=torch.Size((num_classes,)),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
# we let the DirichletClassificationLikelihood compute the targets for us
likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 500
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, likelihood.transformed_targets).sum()
    loss.backward()
    if i % 5 == 0:
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.mean().item(),
            model.likelihood.second_noise_covar.noise.mean().item()
        ))
    optimizer.step()

# prediction
model.eval()
likelihood.eval()

with gpytorch.settings.fast_pred_var(), torch.no_grad():
    test_dist = model(test_x)

    pred_means = test_dist.loc

# plots
os.makedirs('./results', exist_ok=True)

# logits
fig, axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    sub_figs[i]=plt.contourf(
        test_x_mat.numpy(), test_y_mat.numpy(), pred_means[i].numpy().reshape((n_test,n_test))
    )
    ax.set_title("Logits: Class " + str(i), fontsize = 20)
    ax.set_aspect('auto')
    plt.axis([-3, 3, -3, 3])
# set color bar
# cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(sub_fig, cax=cax, **kw)
sys.path.append('../')
from util.common_colorbar import common_colorbar
fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.savefig('./results/cls_GP_logits.png',bbox_inches='tight')


# boundaries
fig = plt.figure(figsize=(5, 5))
plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), pred_means.max(0)[1].reshape((n_test,n_test)))
plt.title('GP', fontsize=20)
plt.savefig('./results/cls_GP_boundaries.png',bbox_inches='tight')