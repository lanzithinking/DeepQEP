"Deep Sigma Point Process Classification Model"

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
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood
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

# testing data
n_test = 50
test_d1 = np.linspace(-3, 3, n_test)
test_d2 = np.linspace(-3, 3, n_test)

test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)
test_labels = torch.round(genfn(test_x_mat, test_y_mat))
test_y = test_labels.view(-1)

# # plot data with true boundary
# sys.path.append('../')
# from util.gpmkr_scatter import gpmkr_scatter
# os.makedirs('./results', exist_ok=True)
# mkrs = ['^', 'o', 'v']
# fig = plt.figure(figsize=(5, 5))
# plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
# gpmkr_scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), m = [mkrs[int(i)] for i in train_y], facecolors='none', edgecolors='r')
# plt.title('True Labels', fontsize=20)
# plt.savefig('./results/cls_truth.png',bbox_inches='tight')
# plt.show()


# Here's a simple standard layer
class DSPPHiddenLayer(DSPPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', Q=8):
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

        super().__init__(variational_strategy, input_dims, output_dims, Q)
        self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(input_dims)}[mean_type]
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None, 
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# define the main model
num_hidden_dspp_dims = 2
num_quadrature_sites = 8

class DirichletDSPP(DSPP):
    def __init__(self, train_x, train_y, likelhood, num_classes):
        hidden_layer = DSPPHiddenLayer(
            input_dims=train_x.shape[-1],
            output_dims=num_hidden_dspp_dims,
            mean_type='linear'
        )
        last_layer = DSPPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_classes,
            mean_type='constant'
        )

        super().__init__(num_quadrature_sites)

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
likelihood.has_global_noise = True
model = DirichletDSPP(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)


# Find optimal model hyperparameters
model.train()
# likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0))

training_iter = 500
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, model.likelihood.transformed_targets.T).sum()
    loss.backward()
    if i % 5 == 0:
        print('Iter %d/%d - Loss: %.3f  hiddenlayer lengthscale: %.3f   noise: %.3f   accuracy: %.4f' % (
            i + 1, training_iter, loss.item(),
            model.hidden_layer.covar_module.base_kernel.lengthscale.mean().item(),
            model.likelihood.second_noise_covar.noise.mean().item(),
            output.mean.mean(0).argmax(-1).eq(train_y).mean(dtype=float).item()
        ))
    optimizer.step()

# prediction
model.eval()
# likelihood.eval()

with gpytorch.settings.fast_pred_var(), torch.no_grad():
    test_dist = model(test_x)

    pred_means = test_dist.mean.mean(0)

acc = pred_means.argmax(-1).eq(test_y).mean(dtype=float).item()
print('Test set: Accuracy: {}%'.format(100. * acc))

# # plots
# os.makedirs('./results', exist_ok=True)
#
# # logits
# fig, axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
# sub_figs = [None]*len(axes.flat)
# for i,ax in enumerate(axes.flat):
#     plt.axes(ax)
#     sub_figs[i]=plt.contourf(
#         test_x_mat.numpy(), test_y_mat.numpy(), pred_means[:,i].numpy().reshape((n_test,n_test))
#     )
#     ax.set_title("Logits: Class " + str(i), fontsize = 20)
#     ax.set_aspect('auto')
#     plt.axis([-3, 3, -3, 3])
# # set color bar
# # cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
# # plt.colorbar(sub_fig, cax=cax, **kw)
# sys.path.append('../')
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
# plt.subplots_adjust(wspace=0.1, hspace=0.2)
# plt.savefig('./results/cls_DSPP_logits.png',bbox_inches='tight')
#
#
# # boundaries
# fig = plt.figure(figsize=(5, 5))
# plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), pred_means.max(1)[1].reshape((n_test,n_test)))
# plt.title('DSPP', fontsize=20)
# plt.savefig('./results/cls_DSPP_boundaries.png',bbox_inches='tight')