"Deep Q-Exponential Process Classification Model"

import os, argparse
import math
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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialDirichletClassificationLikelihood


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
    torch.manual_seed(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    POWER = torch.tensor(1.5, device=device)
    
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
    # X = 2 * (X / X.max(0)[0]) - 1
    # # X = (X - X.mean(0))/X.std(0)
    # y = torch.tensor(data.target.values).long()
    
    # split data
    train_id = np.random.default_rng(2024).choice(len(X), int(np.floor(0.8 * len(X))), replace=False)
    test_id = np.setdiff1d(np.arange(len(X)), train_id)
    
    train_x = X[train_id, :].contiguous().to(device)
    train_y = y[train_id].contiguous().to(device)
    
    test_x = X[test_id, :].contiguous().to(device)
    test_y = y[test_id].contiguous().to(device)
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
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
            return MultivariateQExponential(mean_x, covar_x, power=self.power)
    
    # define the main model
    hidden_features = [10]
    
    class DirichletDeepQEP(DeepQEP):
        def __init__(self, in_features, out_features, hidden_features=2, likelihood=None):
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
            self.likelihood = likelihood
    
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
    
    
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = MultitaskQExponentialDirichletClassificationLikelihood(train_y, learn_additional_noise=True, power=torch.tensor(POWER))
    likelihood.has_global_noise = True
    def _prepare_targets(targets, alpha_epsilon= 0.01, dtype=torch.float, num_classes=None):
            if num_classes is None:
                num_classes = int(targets.max() + 1)
            # set alpha = \alpha_\epsilon
            alpha = alpha_epsilon * torch.ones(targets.shape[-1], num_classes, device=targets.device, dtype=dtype)
    
            # alpha[class_labels] = 1 + \alpha_\epsilon
            alpha[torch.arange(len(targets)), targets] = alpha[torch.arange(len(targets)), targets] + 1.0
    
            # sigma^2 = log(1 / alpha + 1)
            sigma2_i = torch.log(alpha.reciprocal() + 1.0)
    
            # y = log(alpha) - 0.5 * sigma^2
            transformed_targets = alpha.log() - 0.5 * sigma2_i
    
            return sigma2_i.transpose(-2, -1).type(dtype), transformed_targets.type(dtype), num_classes
    likelihood._prepare_targets = _prepare_targets
    model = DirichletDeepQEP(in_features=train_x.shape[-1], out_features=likelihood.num_classes, hidden_features=hidden_features, likelihood=likelihood)
    # set device
    model = model.to(device)
    
    # Find optimal model hyperparameters
    model.train()
    # likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes QExponentialLikelihood parameters
    
    # "Loss" for QEPs - the marginal log likelihood
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
    # mll = DeepLikelihood(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
    
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
            correct_i += output.mean.mean(0).argmax(-1).eq(y_batch).cpu().sum()
            # Calc loss and backprop gradients
            loss = -mll(output, model.likelihood._prepare_targets(y_batch, num_classes=likelihood.num_classes)[1]).sum()
            loss.backward()
            # if i % 5 == 0:
            #     print('Iter %d/%d - Loss: %.3f last layer lengthscale: %.3f   noise: %.3f   accuracy: %.4f' % (
            #         i + 1, num_epochs, loss.item(),
            #         model.layers[-1].covar_module.base_kernel.lengthscale.mean().item(),
            #         model.likelihood.second_noise_covar.noise.mean().item(),
            #         output.mean.mean(0).argmax(-1).eq(train_y).mean(dtype=float).item()
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
    
    from sklearn.metrics import roc_auc_score
    correct = 0
    lls = []
    aucs = []
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        for x_batch, y_batch in test_loader:
            if torch.cuda.is_available():
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            test_dist = model(x_batch)
            pred_means = test_dist.mean.mean(0)
            col_max, pred = torch.max(pred_means,-1)
            correct += pred.eq(y_batch).cpu().sum()
            lls.append(model.likelihood.log_marginal(y_batch[:,None], test_dist).mean(0).cpu())
            y_score = torch.exp(pred_means[:,torch.unique(y_batch)] - col_max[:,None]) if model.likelihood.num_classes>2 else pred_means[torch.arange(y_batch.size(0)),pred][:,None]
            if y_score.size(1)>1: y_score /= y_score.sum(1,keepdims=True)
            if model.likelihood.num_classes>2:
                aucs.append(roc_auc_score(y_batch.cpu(), y_score.cpu(), multi_class='ovo'))
            else:
                aucs.append(roc_auc_score(y_batch.cpu(), y_score.cpu()))
    lls = torch.cat(lls, 0);  aucs = np.stack(aucs)
    NLL = -lls.mean().item()
    ACC = correct / float(len(test_loader.dataset))
    AUC = aucs.mean().item()
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([ACC, AUC, NLL, time_])
    stats = np.array([seed,'DeepQEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'NLL', 'time']
    f_name = os.path.join('./results',args.dataset_name+'_DQEP_'+str(model.num_layers)+'layers.txt')
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