"Deep Q-Exponential Process Model"

import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialLikelihood

POWER = 1.0

# Setting manual seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)

# generate data
train_x = torch.linspace(0, 1, 100)

f = {'step': lambda ts: torch.tensor([1*(t>=0 and t<=1) + 0.5*(t>1 and t<=1.5) + 2*(t>1.5 and t<=2) for t in ts]),
     'turning': lambda ts: torch.tensor([1.5*t*(t>=0 and t<=1) + (3.5-2*t)*(t>1 and t<=1.5) + (3*t-4)*(t>1.5 and t<=2) for t in ts])}

train_y = torch.stack([
    f['step'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
    f['turning'](train_x * 2) + torch.randn(train_x.size()) * 0.1,
], -1)

train_x = train_x.unsqueeze(-1)


# Here's a simple standard layer
class DQEPLayer(DeepQEPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        self.power = torch.tensor(POWER)
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
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
        self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(input_dims)}[mean_type]
        # self.covar_module = ScaleKernel(
        #     RBFKernel(
        #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
        #             np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
        #         ), batch_shape=batch_shape, ard_num_dims=input_dims
        #     )
        # )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
            ),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.power)

# define the main model
num_tasks = train_y.size(-1)
hidden_features = [3]


class MultitaskDeepQEP(DeepQEP):
    def __init__(self, in_features, out_features, hidden_features=2):
        super().__init__()
        if isinstance(hidden_features, int):
            layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
        elif isinstance(hidden_features, list):
            layer_config = [in_features]+hidden_features+[out_features]
        layers = []
        for i in range(len(layer_config)-1):
            layers.append(DQEPLayer(
                input_dims=layer_config[i],
                output_dims=layer_config[i+1],
                mean_type='linear' if i < len(layer_config)-2 else 'constant'
            ))
        self.num_layers = len(layers)
        self.layers = torch.nn.Sequential(*layers)
        # We're going to use a multitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=out_features, power=torch.tensor(POWER))

    def forward(self, inputs):
        output = self.layers[0](inputs)
        for i in range(1,len(self.layers)):
            output = self.layers[i](output)
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


model = MultitaskDeepQEP(in_features=train_x.shape[-1], out_features=num_tasks, hidden_features=hidden_features)


# training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

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
# truth
test_y = torch.stack([
    f['step'](test_x * 2),
    f['turning'](test_x * 2),
], -1)
MAE = torch.mean(torch.abs(mean-test_y))
RMSE = torch.mean(torch.pow(mean-test_y, 2)).sqrt()
print('Test MAE: {}'.format(MAE))
print('Test RMSE: {}'.format(RMSE))

# Plot results
fig, axs = plt.subplots(1, num_tasks+1, figsize=(4 * (num_tasks+1), 4))
for task, ax in enumerate(axs):
    if task < num_tasks:
        ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
        ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
        ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
        ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
        ax.set_ylim([-.5, 3])
        ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'], fontsize=12)
        ax.set_title(f'Task {task + 1}: '+list(f.keys())[task]+' function', fontsize=20)
    else:
        ax.plot(loss_list)
        ax.set_title('Neg. ELBO Loss', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
fig.tight_layout()

# plt.show()
os.makedirs('./results', exist_ok=True)
plt.savefig('./results/ts_DeepQEP_'+str(model.num_layers)+'layers.png',bbox_inches='tight')