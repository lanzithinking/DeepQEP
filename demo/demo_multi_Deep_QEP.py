"Showcase Deep Q-Exponential Process Model"

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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialLikelihood

POWER = 1.0

# Setting manual seed for reproducibility
torch.manual_seed(73)
# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

# generate data
train_x = torch.linspace(0, 1, 100)

train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    torch.sin(train_x * (2 * math.pi)) + 2 * torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
], -1)

train_x = train_x.unsqueeze(-1)


# Here's a simple standard layer
class DQEPHiddenLayer(DeepQEPLayer):
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
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateQExponential(mean_x, covar_x, power=self.power)

# define the main model
num_tasks = train_y.size(-1)
num_hidden_dqep_dims = 3


class MultitaskDeepQEP(DeepQEP):
    def __init__(self, train_x_shape):
        hidden_layer = DQEPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dqep_dims,
            mean_type='linear'
        )
        last_layer = DQEPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskQExponentialLikelihood(num_tasks=num_tasks, power=torch.tensor(POWER))

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


model = MultitaskDeepQEP(train_x.shape)


# training
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

num_epochs = 1 if smoke_test else 200
epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()

# Make predictions
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    mean, var = model.predict(test_x)
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

# Plot results
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
for task, ax in enumerate(axs):
    ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
    ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(f'Task {task + 1}')
fig.tight_layout()

# plt.show()
plt.savefig('./demo_multi_DeepQEP.png',bbox_inches='tight')