"Q-Exponential Process Model"

import os
import random
import numpy as np
import timeit
from matplotlib import pyplot as plt

import torch
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models import ExactQEP
from gpytorch.likelihoods import MultitaskQExponentialLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel

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

def main(seed=2024):
    
    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Define model
    # We will use the simplest form of QEP model, exact inference
    class MultitaskQEPModel(ExactQEP):
        def __init__(self, train_x, train_y, num_tasks):
            likelihood = MultitaskQExponentialLikelihood(num_tasks=num_tasks, power=torch.tensor(POWER), miu=False)
            super(MultitaskQEPModel, self).__init__(train_x, train_y, likelihood)
            # self.mean_module = ConstantMean(batch_shape=torch.Size((num_tasks,)))
            # # self.covar_module = ScaleKernel(
            # #     RBFKernel(batch_shape=torch.Size((num_tasks,))),
            # #     batch_shape=torch.Size((num_tasks,)),
            # # )
            # self.covar_module = ScaleKernel(
            #     MaternKernel(nu=1.5, batch_shape=torch.Size((num_tasks,)), ard_num_dims=train_x.shape[-1],
            #         # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #         #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
            #     ),
            #     batch_shape=torch.Size((num_tasks,)),
            # )
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
            # self.covar_module = gpytorch.kernels.MultitaskKernel(
            #     gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, #rank=1
            # )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(nu=1.5), num_tasks=num_tasks, #rank=1
            )
            self.likelihood = likelihood
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateQExponential(mean_x, covar_x, power=torch.tensor(POWER))
        
        def predict(self, test_x):
            with torch.no_grad():
    
                # The output of the model is a multitask QEP, where both the data points
                # and the tasks are jointly distributed
                output = self(test_x)
                # To compute the marginal predictive NLL of each data point,
                # we will call `to_data_independent_dist`,
                # which removes the data cross-covariance terms from the distribution.
                preds = self.likelihood(output).to_data_independent_dist()
    
            return preds.mean, preds.variance, output
    
    # initialize model
    num_tasks = train_y.size(-1)
    model = MultitaskQEPModel(train_x, train_y, num_tasks=num_tasks)
    
    
    # Find optimal model hyperparameters
    model.train()
    # likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes QExponentialLikelihood parameters
    
    # "Loss" for QEPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    
    loss_list = []
    num_epochs = 1000
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning=timeit.default_timer()
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
    time_ = timeit.default_timer()-beginning
    print('Training uses: {} seconds.'.format(time_))
    
    # prediction
    model.eval()
    # likelihood.eval()
    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # preds = likelihood(model(test_x))
        # mean, var = preds.mean, preds.variance
        mean, var, mdl_output = model.predict(test_x)
        lower = mean - 2 * var.sqrt()
        upper = mean + 2 * var.sqrt()
    # truth
    test_y = torch.stack([
        f['step'](test_x * 2),
        f['turning'](test_x * 2),
    ], -1)
    MAE = torch.mean(torch.abs(mean-test_y)).item()
    RMSE = torch.mean(torch.pow(mean-test_y, 2)).sqrt().item()
    PSD = torch.mean(var.sqrt()).item()
    NLL = -model.likelihood.log_marginal(test_y, mdl_output).mean(0).mean().item()
    from sklearn.metrics import r2_score
    R2 = r2_score(test_y, mean)
    print('Test MAE: {}'.format(MAE))
    print('Test RMSE: {}'.format(RMSE))
    print('Test PSD: {}'.format(PSD))
    print('Test R2: {}'.format(R2))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([MAE, RMSE, PSD, R2, NLL, time_])
    stats = np.array([seed,'QEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
    f_name = os.path.join('./results/ts_QEP.txt')
    with open(f_name,'ab') as f_:
        np.savetxt(f_,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')
    
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
    plt.savefig('./results/ts_QEP.png',bbox_inches='tight')

if __name__ == '__main__':
    main()
    # n_seed = 10; i=0; n_success=0; n_failure=0
    # while n_success < n_seed and n_failure < 10* n_seed:
    #     seed_i=2024+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         n_failure+=1
    #         pass
    #     i+=1