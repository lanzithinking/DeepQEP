"Deep Sigma Point Process Model"

import os, argparse
import math
import numpy as np
import timeit

import torch
import tqdm
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# prepare UCI regression datasets
if not os.path.exists('./uci_datasets'):
    os.system('/user/local/bin/python3 -m pip install git+https://github.com/treforevans/uci_datasets.git --target=./uci_datasets')
sys.path.append('./uci_datasets')
from uci_datasets import Dataset

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='elevators')
    args = parser.parse_args()
    
    # Setting manual seed for reproducibility
    torch.manual_seed(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = Dataset(args.dataset_name)
    
    X = torch.Tensor(data.x)
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = torch.Tensor(data.y)
    
    # split data
    train_n = int(math.floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous().to(device)
    train_y = y[:train_n, :].contiguous().to(device)
    
    test_x = X[train_n:, :].contiguous().to(device)
    test_y = y[train_n:, :].contiguous().to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    
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
            # self.covar_module = ScaleKernel(
            #     RBFKernel(
            #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            #         ), batch_shape=batch_shape, ard_num_dims=input_dims
            #     )
            # )
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims),
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
    num_tasks = train_y.size(-1)
    num_hidden_dspp_dims = 3
    num_quadrature_sites = 8
    
    
    class MultitaskDSPP(DSPP):
        def __init__(self, train_x_shape):
            hidden_layer = DSPPHiddenLayer(
                input_dims=train_x_shape[-1],
                output_dims=num_hidden_dspp_dims,
                mean_type='linear'
            )
            last_layer = DSPPHiddenLayer(
                input_dims=hidden_layer.output_dims,
                output_dims=num_tasks,
                mean_type='constant'
            )
    
            super().__init__(num_quadrature_sites)
    
            self.hidden_layer = hidden_layer
            self.last_layer = last_layer
    
            # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
    
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
    
    
    model = MultitaskDSPP(train_x.shape)
    # set device
    model = model.to(device)
    
    # training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=train_y.size(0))
    
    
    loss_list = []
    num_epochs = 1000
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning=timeit.default_timer()
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        loss_i = 0
        for x_batch, y_batch in minibatch_iter:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            loss_i += loss.item()
        loss_i /= len(minibatch_iter)
        # record the loss and the best model
        loss_list.append(loss_i)
        if i==0:
            min_loss = loss_list[-1]
            optim_model = model.state_dict()
        else:
            if loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
    time_ = timeit.default_timer()-beginning
    print('Training uses: {} seconds.'.format(time_))
    
    # load the best model
    model.load_state_dict(optim_model)
    # Make predictions
    model.eval()
    means = []
    vars = []
    lls = []
    with torch.no_grad():#, gpytorch.settings.fast_pred_var():
        for x_batch, y_batch in test_loader:
            # mean, var = model.predict(x_batch)
            # means = torch.cat([means, mean.cpu()])
            # vars = torch.cat([means, var.cpu()])
            preds = model(x_batch)
            means.append(preds.mean.mean(0).cpu())
            vars.append(preds.variance.mean(0).cpu())
            lls.append(model.likelihood.log_marginal(y_batch, preds).mean(0).cpu())
    means = torch.cat(means, 0)
    vars = torch.cat(vars, 0)
    lls = torch.cat(lls, 0)
    
    MAE = torch.mean(torch.abs(means - test_y.cpu())).item()
    RMSE = torch.mean(torch.pow(means - test_y.cpu(), 2)).sqrt().item()
    STD = torch.mean(vars.sqrt()).item()
    NLL = -lls.mean().item()
    print('Test MAE: {}'.format(MAE))
    print('Test RMSE: {}'.format(RMSE))
    print('Test std: {}'.format(STD))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([MAE, RMSE, STD, NLL, time_])
    stats = np.array(['DSPP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['Method', 'MAE', 'RMSE', 'STD', 'NLL', 'time']
    np.savetxt('./results/elevators_DSPP.txt',stats,fmt="%s",delimiter=',',header=','.join(header))

if __name__ == '__main__':
    main()