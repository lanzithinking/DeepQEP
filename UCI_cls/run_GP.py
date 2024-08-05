"Gaussian Process Classification Model"

import os, argparse
import torch
import numpy as np
import timeit

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel

# prepare UCI classification datasets
if not os.path.exists('./uci_dataset'):
    os.system('/usr/local/bin/python3 -m pip install git+https://github.com/maryami66/uci_dataset.git --target=./uci_dataset')
sys.path.append('./uci_dataset')
import uci_dataset as dataset

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='heart_disease')
    args = parser.parse_args()
    
    # Setting manual seed for reproducibility
    torch.manual_seed(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    # load data set
    data = getattr(dataset,'load_'+args.dataset_name)().dropna()
    
    X = torch.tensor(data.loc[:,data.columns!='target'].values).float()
    # X = X - X.min(0)[0]
    # X = 2 * (X / X.max(0)[0]) - 1
    X = (X - X.mean(0))/X.std(0)
    y = torch.tensor(data.target.values).long()
    
    # split data
    train_id = np.random.default_rng(2024).choice(len(data), int(np.floor(0.75 * len(data))), replace=False)
    test_id = np.setdiff1d(np.arange(len(data)), train_id)
    
    train_x = X[train_id, :].contiguous().to(device)
    train_y = y[train_id].contiguous().to(device)
    
    test_x = X[test_id, :].contiguous().to(device)
    test_y = y[test_id].contiguous().to(device)
    
    # train_dataset = TensorDataset(train_x, train_y)
    # train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    #
    # test_dataset = TensorDataset(test_x, test_y)
    # test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Define model
    # We will use the simplest form of GP model, exact inference
    class DirichletGPModel(ExactGP):
        def __init__(self, train_x, train_y, likelihood, num_classes):
            super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=torch.Size((num_classes,))),
                batch_shape=torch.Size((num_classes,)),
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # initialize likelihood and model
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
    
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    loss_list = []
    num_epochs = 1000
    beginning=timeit.default_timer()
    for i in range(num_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        if i % 5 == 0:
            print('Iter %d/%d - Loss: %.3f lengthscale: %.3f   noise: %.3f   accuracy: %.4f' % (
                i + 1, num_epochs, loss.item(),
                model.covar_module.base_kernel.lengthscale.mean().item(),
                likelihood.second_noise_covar.noise.mean().item(),
                output.mean.argmax(0).eq(train_y).mean(dtype=float).item()
            ))
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
    likelihood.eval()
    
    from sklearn.metrics import roc_auc_score
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)
        pred_means = test_dist.mean
        col_max, pred = torch.max(pred_means,0)
        lls = likelihood.log_marginal(test_y, test_dist).mean(0).cpu()
        y_score = torch.exp(pred_means - col_max)
        y_score /= y_score.sum(0)
    NLL = -lls.mean().item()
    ACC = pred.eq(test_y).mean(dtype=float).item()
    AUC = roc_auc_score(test_y, y_score.T, multi_class='ovo').item()
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([ACC, AUC, NLL, time_])
    stats = np.array([seed,'GP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'NLL', 'time']
    f_name = os.path.join('./results',args.dataset_name+'_GP.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')

if __name__ == '__main__':
    main()
    # n_seed = 10; i=0; n_success=0
    # while n_success < n_seed:
    #     seed_i=2024+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         pass
    #     i+=1