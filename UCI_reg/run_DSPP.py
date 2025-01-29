"Deep Sigma Point Process Model"

import os, argparse
import random
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
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = Dataset(args.dataset_name)
    
    X = torch.Tensor(data.x)
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = torch.Tensor(data.y)
    
    # split data
    train_n = int(np.floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()#.to(device)
    train_y = y[:train_n, :].contiguous()#.to(device)
    
    test_x = X[train_n:, :].contiguous()#.to(device)
    test_y = y[train_n:, :].contiguous()#.to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    
    # Here's a simple standard layer
    class DSPP_Layer(DSPPLayer):
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
            #             np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
            #         ), batch_shape=batch_shape, ard_num_dims=input_dims
            #     )
            # )
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                    # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
                ),
                batch_shape=batch_shape, ard_num_dims=None
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)
    
    # define the main model
    class MultitaskDSPP(DSPP):
        def __init__(self, in_features, out_features, hidden_features=2, num_quadrature_sites=8):
            super().__init__(num_quadrature_sites)
            if isinstance(hidden_features, int):
                layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
            elif isinstance(hidden_features, list):
                layer_config = [in_features]+hidden_features+[out_features]
            layers = []
            for i in range(len(layer_config)-1):
                layers.append(DSPP_Layer(
                    input_dims=layer_config[i],
                    output_dims=layer_config[i+1],
                    mean_type='linear' if i < len(layer_config)-2 else 'constant',
                    Q = num_quadrature_sites
                ))
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
            # We're going to use a multitask likelihood instead of the standard GaussianLikelihood
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=out_features)
    
        def forward(self, inputs):
            output = self.layers[0](inputs)
            for i in range(1,len(self.layers)):
                output = self.layers[i](output)
            return output
    
        def predict(self, test_loader):
            means, vars, lls = [], [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    if torch.cuda.is_available():
                        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                    output = self(x_batch)
                    preds = self.likelihood(output)
                    weights = self.quad_weights
                    means.append((weights.exp()[:,None,None] * preds.mean).sum(0).cpu())
                    vars.append(preds.variance.mean(0).cpu())
                    lls.append((self.likelihood.log_marginal(y_batch, output)+weights.unsqueeze(-1)).logsumexp(dim=0).cpu())
            return torch.cat(means, 0), torch.cat(vars, 0), torch.cat(lls, 0)
    
    num_tasks = train_y.size(-1)
    hidden_features = [3]
    num_quadrature_sites = 8
    model = MultitaskDSPP(in_features=train_x.shape[-1], out_features=num_tasks, hidden_features=hidden_features, num_quadrature_sites=num_quadrature_sites)
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
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
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
    with torch.no_grad(), gpytorch.settings.fast_computations(log_prob=False, solves=False):
        means, vars, lls = model.predict(test_loader)
    
    MAE = torch.mean(torch.abs(means - test_y.cpu())).item()
    RMSE = torch.mean(torch.pow(means - test_y.cpu(), 2)).sqrt().item()
    PSD = torch.mean(vars.sqrt()).item()
    NLL = -lls.mean().item()
    from sklearn.metrics import r2_score
    R2 = r2_score(test_y.cpu(), means).mean()
    print('Test MAE: {}'.format(MAE))
    print('Test RMSE: {}'.format(RMSE))
    print('Test PSD: {}'.format(PSD))
    print('Test R2: {}'.format(R2))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([MAE, RMSE, PSD, R2, NLL, time_])
    stats = np.array([seed,'DSPP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
    f_name = os.path.join('./results',args.dataset_name+'_DSPP_'+str(model.num_layers)+'layers.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0; n_failure=0
    while n_success < n_seed and n_failure < 10* n_seed:
        seed_i=2024+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            n_failure+=1
            pass
        i+=1