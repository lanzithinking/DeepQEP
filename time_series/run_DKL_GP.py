"Deep Kernel Learning Gaussian Process Model"

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
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# Setting manual seed for reproducibility
# seed=202408
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # define the NN feature extractor
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
    
    data_dim = train_x.size(-1)
    num_features = train_y.size(-1)
    hidden_features = [100, 50]
    feature_extractor = FeatureExtractor(in_features=data_dim, out_features=num_features, hidden_features=hidden_features)
    
    
    # define the GP layer
    class GaussianProcessLayer(ApproximateGP):
        def __init__(self, input_dims, output_dims, grid_bounds=(-10., 10.), grid_size=64, mean_type='constant'):
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
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return MultivariateNormal(mean, covar)
    
    
    # define the main model
    class MultitaskDKLGP(gpytorch.Module):
        def __init__(self, feature_extractor, output_dims, grid_bounds=(-10., 10.)):
            super(MultitaskDKLGP, self).__init__()
            self.feature_extractor = feature_extractor
            self.gp_layer = GaussianProcessLayer(input_dims=num_features, output_dims=output_dims, grid_bounds=grid_bounds)
            
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims)
    
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
                output = self(test_x)
                # To compute the marginal predictive NLL of each data point,
                # we will call `to_data_independent_dist`,
                # which removes the data cross-covariance terms from the distribution.
                preds = self.likelihood(output).to_data_independent_dist()
    
            # return preds.mean.mean(0), preds.variance.mean(0)
            return preds.mean, preds.variance, output
    
    num_tasks = train_y.size(-1)
    model = MultitaskDKLGP(feature_extractor, output_dims=num_tasks)
    
    
    # training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = VariationalELBO(model.likelihood, model.gp_layer, num_data=train_y.size(0))
    
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
    
    # load the best model
    # model.load_state_dict(optim_model)
    # Make predictions
    model.eval()
    test_x = torch.linspace(0, 1, 51).unsqueeze(-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
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
    stats = np.array([seed,'DKLGP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'MAE', 'RMSE', 'PSD', 'R2', 'NLL', 'time']
    f_name = os.path.join('./results/ts_DKLGP_'+str(model.feature_extractor.num_layers)+'layers.txt')
    with open(f_name,'ab') as f_:
        np.savetxt(f_,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')
    
    # # Plot results
    # fig, axs = plt.subplots(1, num_tasks+1, figsize=(4 * (num_tasks+1), 4))
    # for task, ax in enumerate(axs):
    #     if task < num_tasks:
    #         ax.plot(test_x.squeeze(-1).numpy(), list(f.values())[task](test_x*2).numpy(), 'r--')
    #         ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    #         ax.plot(test_x.squeeze(-1).numpy(), mean[:, task].numpy(), 'b')
    #         ax.fill_between(test_x.squeeze(-1).numpy(), lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    #         ax.set_ylim([-.5, 3])
    #         ax.legend(['Truth','Observed Data', 'Mean', 'Confidence'], fontsize=12)
    #         ax.set_title(f'Task {task + 1}: '+list(f.keys())[task]+' function', fontsize=20)
    #     else:
    #         ax.plot(loss_list)
    #         ax.set_title('Neg. ELBO Loss', fontsize=20)
    #     ax.tick_params(axis='both', which='major', labelsize=14)
    # fig.tight_layout()
    #
    # # plt.show()
    # os.makedirs('./results', exist_ok=True)
    # plt.savefig('./results/ts_DKLGP_'+str(model.feature_extractor.num_layers)+'layers.png',bbox_inches='tight')

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