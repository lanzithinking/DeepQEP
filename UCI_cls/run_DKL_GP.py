"Deep Kernel Learning Gaussian Process Classification Model"

import os, argparse
import random
import numpy as np
import timeit
from sklearn.metrics import roc_auc_score

import torch
import tqdm
from torch.utils.data import TensorDataset, DataLoader

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy, GridInterpolationVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskDirichletClassificationLikelihood


# prepare UCI classification datasets
# if not os.path.exists('./uci_dataset'):
#     os.system('/usr/local/bin/python3 -m pip install git+https://github.com/maryami66/uci_dataset.git --target=./uci_dataset')
# sys.path.append('./uci_dataset')
# import uci_dataset as dataset
from uci_utils import UCI_Dataset_Loader as dataset

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='car')
    args = parser.parse_args()
    
    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    # load data set
    # data = getattr(dataset,'load_'+args.dataset_name)().dropna()
    X, y = getattr(dataset,args.dataset_name)()
    try:
        X = torch.tensor(X.values).float()
    except:
        X = torch.Tensor(X.values.astype(float))
    y = torch.tensor(y.values).long()
    
    # X = torch.tensor(data.loc[:,data.columns!='target'].values).float()
    # X = X - X.min(0)[0]
    # X = 2 * (X / (X.max(0)[0]+(X.max(0)[0]==0)*torch.ones(X.shape[1:]))) - 1
    # X = (X - X.mean(0))/(X.std(0)+(X.std(0)==0)*torch.ones(X.shape[1:]))
    # y = torch.tensor(data.target.values).long()
    
    # split data
    train_id = np.random.default_rng(2024).choice(len(X), int(np.floor(0.8 * len(X))), replace=False)
    test_id = np.setdiff1d(np.arange(len(X)), train_id)
    
    train_x = X[train_id, :].contiguous()#.to(device)
    train_y = y[train_id].contiguous()#.to(device)
    
    test_x = X[test_id, :].contiguous()#.to(device)
    test_y = y[test_id].contiguous()#.to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
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
    num_features = len(torch.unique(y))
    hidden_features = [1000, 500, 50]
    feature_extractor = FeatureExtractor(in_features=data_dim, out_features=num_features, hidden_features=hidden_features)
    
    
    # define the GP layer
    class GaussianProcessLayer(ApproximateGP):
        def __init__(self, input_dims, output_dims, grid_bounds=(-10., 10.), grid_size=128, mean_type='constant'):
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
            return MultivariateNormal(mean, covar)
    
    
    # define the main model
    class clsDKLGP(gpytorch.Module):
        def __init__(self, feature_extractor, output_dims, likelihood, grid_bounds=(-10., 10.)):
            super(clsDKLGP, self).__init__()
            self.feature_extractor = feature_extractor
            self.gp_layer = GaussianProcessLayer(input_dims=num_features, output_dims=output_dims, grid_bounds=grid_bounds)
            self.likelihood = likelihood
    
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])
    
        def forward(self, x):
            features = self.feature_extractor(x)
            features = self.scale_to_bounds(features)
            # This next line makes it so that we learn a GP for each feature
            features = features.transpose(-1, -2).unsqueeze(-1)
            res = self.gp_layer(features)
            return res
    
        def predict(self, test_loader):
            correct = 0
            lls, aucs = [], []
            with gpytorch.settings.fast_pred_var(), torch.no_grad():
                for x_batch, y_batch in test_loader:
                    if torch.cuda.is_available():
                        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
                    test_dist = model(x_batch)
                    pred_means = test_dist.mean#.mean(0)
                    col_max, pred = torch.max(pred_means,-1)
                    correct += pred.eq(y_batch).cpu().sum()
                    batch_targets = model.likelihood._prepare_targets(y_batch, num_classes=model.likelihood.num_classes)[1]
                    lls.append(model.likelihood.log_marginal(batch_targets, test_dist).cpu())
                    y_score = torch.exp(pred_means[:,torch.unique(y_batch)] - col_max[:,None]) if model.likelihood.num_classes>2 else pred_means[torch.arange(y_batch.size(0)),pred][:,None]
                    if y_score.size(1)>1: y_score /= y_score.sum(1,keepdims=True)
                    if model.likelihood.num_classes>2:
                        aucs.append(roc_auc_score(y_batch.cpu(), y_score.cpu(), multi_class='ovo'))
                    else:
                        aucs.append(roc_auc_score(y_batch.cpu(), y_score.cpu()))
            return correct, np.stack(aucs), torch.cat(lls, 0)
    
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = MultitaskDirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    likelihood.has_global_noise = True
    model = clsDKLGP(feature_extractor, likelihood.num_classes, likelihood)
    # set device
    model = model.to(device)
    
    # Find optimal model hyperparameters
    model.train()
    # likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = VariationalELBO(model.likelihood, model.gp_layer, num_data=train_y.size(0))
    
    loss_list = []
    num_epochs = 1000
    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning=timeit.default_timer()
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        correct_i = 0
        loss_i = 0
        for x_batch, y_batch in minibatch_iter:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_batch)
            correct_i += output.mean.argmax(-1).eq(y_batch).cpu().sum()
            # Calc loss and backprop gradients
            loss = -mll(output, model.likelihood._prepare_targets(y_batch, num_classes=model.likelihood.num_classes)[1]).sum()
            loss.backward()
            # if i % 5 == 0:
            #     print('Iter %d/%d - Loss: %.3f last layer lengthscale: %.3f   noise: %.3f   accuracy: %.4f' % (
            #         i + 1, num_epochs, loss.item(),
            #         model.layers[-1].covar_module.base_kernel.lengthscale.mean().item(),
            #         model.likelihood.second_noise_covar.noise.mean().item(),
            #         output.mean.argmax(-1).eq(y_batch).mean(dtype=float).item()
            #     ))
            optimizer.step()
            loss_i += loss.item()
            minibatch_iter.set_postfix(loss=loss.item())
        loss_i /= len(minibatch_iter)
        acc = correct_i / float(len(train_loader.dataset))
        epochs_iter.set_postfix(acc=acc.item())
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
    # prediction
    model.eval()
    # likelihood.eval()
    
    # from sklearn.metrics import roc_auc_score
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        correct, aucs, lls = model.predict(test_loader)
    NLL = -lls.mean().item()
    ACC = correct / float(len(test_loader.dataset))
    AUC = aucs.mean().item()
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([ACC, AUC, NLL, time_])
    stats = np.array([seed,'DKLGP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'NLL', 'time']
    f_name = os.path.join('./results',args.dataset_name+'_DKLGP_'+str(model.feature_extractor.num_layers)+'layers.txt')
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