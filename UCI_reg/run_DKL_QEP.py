"Deep Kernel Learning Q-Exponential Process Model"

import os, argparse
import numpy as np
import timeit

import torch
import tqdm
from torch.utils.data import TensorDataset, DataLoader

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import CholeskyVariationalDistribution, UncorrelatedMultitaskVariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models import ApproximateQEP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialLikelihood

# prepare UCI regression datasets
if not os.path.exists('./uci_datasets'):
    os.system('/usr/local/bin/python3 -m pip install git+https://github.com/treforevans/uci_datasets.git --target=./uci_datasets')
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
    print('Using the '+device+' device...')
    
    POWER = torch.tensor(1.0, device=device)
    
    # load data set
    data = Dataset(args.dataset_name)
    
    X = torch.Tensor(data.x)
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = torch.Tensor(data.y)
    
    # split data
    train_n = int(np.floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous().to(device)
    train_y = y[:train_n, :].contiguous().to(device)
    
    test_x = X[train_n:, :].contiguous().to(device)
    test_y = y[train_n:, :].contiguous().to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    
    # define the NN feature extractor
    data_dim = train_x.size(-1)
    num_features = train_y.size(-1)
    hidden_features = [1000, 500, 50]
    
    class FeatureExtractor(torch.nn.Sequential):
        def __init__(self, in_features, out_features, hidden_features=2):
            super(FeatureExtractor, self).__init__()
            if isinstance(hidden_features, int):
                layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
            elif isinstance(hidden_features, list):
                layer_config = [in_features]+hidden_features+[out_features]
            layers = []
            for i in range(len(layer_config)-1):
                layers.append(torch.nn.Linear(layer_config[i],layer_config[i+1]))
                if i < len(layer_config)-2:
                    layers.append(torch.nn.ReLU())
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
    
    feature_extractor = FeatureExtractor(in_features=data_dim, out_features=num_features, hidden_features=hidden_features)
    
    
    # define the QEP layer
    class QExponentialProcessLayer(ApproximateQEP):
        def __init__(self, input_dims, output_dims, grid_bounds=(-10., 10.), grid_size=128, mean_type='constant'):
            self.power = POWER
            batch_shape = torch.Size([output_dims])
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_shape=batch_shape, power=self.power
            )
    
            # Our base variational strategy is a GridInterpolationVariationalStrategy,
            # which places variational inducing points on a Grid
            # We wrap it with a UncorrelatedMultitaskVariationalStrategy so that our output is a vector-valued QEP
            variational_strategy = UncorrelatedMultitaskVariationalStrategy(
                GridInterpolationVariationalStrategy(
                    self, grid_size=grid_size, grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ), num_tasks=output_dims,
            )
            super().__init__(variational_strategy)
    
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
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return MultivariateQExponential(mean, covar, power=self.power)
    
    
    # define the main model
    num_tasks = train_y.size(-1)
    
    class MultitaskDKLQEP(gpytorch.Module):
        def __init__(self, feature_extractor, output_dims, grid_bounds=(-10., 10.)):
            super(MultitaskDKLQEP, self).__init__()
            self.feature_extractor = feature_extractor
            self.qep_layer = QExponentialProcessLayer(input_dims=num_features, output_dims=output_dims, grid_bounds=grid_bounds)
            
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])
            self.likelihood = MultitaskQExponentialLikelihood(num_tasks=output_dims, power=POWER)
    
        def forward(self, x):
            features = self.feature_extractor(x)
            features = self.scale_to_bounds(features)
            # This next line makes it so that we learn a QEP for each feature
            features = features.transpose(-1, -2).unsqueeze(-1)
            res = self.qep_layer(features)
            return res
    
        def predict(self, test_x):
            with torch.no_grad():
    
                # The output of the model is a multitask MVN, where both the data points
                # and the tasks are jointly distributed
                # To compute the marginal predictive NLL of each data point,
                # we will call `to_data_independent_dist`,
                # which removes the data cross-covariance terms from the distribution.
                preds = model.likelihood(model(test_x)).to_data_independent_dist()
    
            # return preds.mean.mean(0), preds.variance.mean(0)
            return preds.mean, preds.variance
    
    
    model = MultitaskDKLQEP(feature_extractor, output_dims=num_tasks)
    # set device
    model = model.to(device)
    
    # training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = VariationalELBO(model.likelihood, model.qep_layer, num_data=train_y.size(0))
    
    
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
            means.append(preds.mean.cpu())
            vars.append(preds.variance.cpu())
            lls.append(model.likelihood.log_marginal(y_batch, preds).cpu())
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
    stats = np.array([seed,'DKLQEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'MAE', 'RMSE', 'STD', 'NLL', 'time']
    f_name = os.path.join('./results',args.dataset_name+'_DKLQEP_'+str(model.feature_extractor.num_layers)+'layers.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0
    while n_success < n_seed:
        seed_i=2024+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            pass
        i+=1