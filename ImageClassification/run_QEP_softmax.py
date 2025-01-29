"Q-Exponential Process Classification Model"

import os, argparse
import random
import numpy as np
import timeit
from sklearn.metrics import roc_auc_score

import torch
import tqdm
from torch.optim.lr_scheduler import MultiStepLR

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, UncorrelatedMultitaskVariationalStrategy
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models import ApproximateQEP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from dataset import dataset
os.makedirs('./results', exist_ok=True)

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='mnist')
    parser.add_argument('batch_size', nargs='?', type=int, default=256)
    args = parser.parse_args()
    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    POWER = torch.tensor(1.0, device=device)
    
    # load data
    train_loader, test_loader, feature_extractor, num_classes = dataset(args.dataset_name, seed, batch_size=args.batch_size)
    # input_dim = np.prod(list(train_loader.dataset.data.shape[1:]))
    input_dim = np.prod(next(iter(train_loader))[0].shape[1:])
    
    # Here's a QEP model
    class QEPModel(ApproximateQEP):
        def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant', likelihood=None):
            self.power = POWER
            self.input_dims = input_dims
            self.output_dims = output_dims
            inducing_points = torch.randn(num_inducing, input_dims, device=device)
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
            # variational_strategy = UncorrelatedMultitaskVariationalStrategy(
            #     variational_strategy, num_tasks=output_dims
            # )
    
            super().__init__(variational_strategy)
            self.mean_module = {'constant': ConstantMean(batch_shape=batch_shape), 'linear': LinearMean(input_dims)}[mean_type]
            # self.covar_module = ScaleKernel(
            #     RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims,
            #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #             np.exp(-1), np.exp(1), sigma=.1, transform=torch.exp
            #         )
            #     ),
            #     batch_shape=batch_shape,
            # )
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                    # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
                ),
                batch_shape=batch_shape,
            )
            self.likelihood = likelihood
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateQExponential(mean_x, covar_x, power=self.power)
    
    # define likelihood and model
    likelihood = SoftmaxLikelihood(num_features=num_classes, num_classes=num_classes)#, mixing_weights=False)
    model = QEPModel(input_dims=input_dim, output_dims=num_classes, likelihood=likelihood)
    # set device
    model = model.to(device)
    # for p in model.parameters():
    #     p.data.uniform_(0,1./np.sqrt(input_dim/10))
    # likelihood = likelihood.to(device)
    # for p in likelihood.parameters():
    #     p.data.uniform_(0,1./np.sqrt(input_dim/10))
    
    # "Loss" for QEPs - the marginal log likelihood
    mll = VariationalELBO(model.likelihood, model, num_data=len(train_loader.dataset))
    
    # Use the adam optimizer
    # optimizer = torch.optim.Adam([{'params':model.parameters()},
    #                               {'params':likelihood.parameters()}], lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr = 0.1
    # optimizer = torch.optim.SGD([
    #     {'params': model.hyperparameters(), 'lr': lr * 0.01},
    #     {'params': model.variational_parameters()},
    #     {'params': likelihood.parameters()},
    # ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    num_epochs = 1000
    scheduler = MultiStepLR(optimizer, milestones=[0.25 * num_epochs, 0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)

    # define training and testing procedures
    def train(epoch):
        model.train()
        # likelihood.train()
    
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        with gpytorch.settings.num_likelihood_samples(8):
            for data, target in minibatch_iter:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data.reshape((data.shape[0],-1)))
                loss = -mll(output, target).sum()
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())
        return loss.item()
    
    def test(epoch):
        model.eval()
        # likelihood.eval()
    
        correct = 0
        lls, aucs = [], []
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                appx_dist = model(data.reshape((data.shape[0],-1)))
                output = likelihood(appx_dist)  # This gives us 16 samples from the predictive distribution
                y_score = output.probs.mean(0)
                pred = y_score.argmax(-1)  # Taking the mean over all of the sample we've drawn
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                lls.append(likelihood.log_marginal(target, appx_dist).cpu())
                aucs.append(roc_auc_score(target.cpu(), y_score.cpu(), multi_class='ovo'))
        aucs = np.stack(aucs); lls = torch.cat(lls, 0)
        acc = correct / float(len(test_loader.dataset))
        auc = aucs.mean().item()
        nll = -lls.mean().item()
        print('Epoch {}: test set: Accuracy: {}/{} ({}%), AUC: {}'.format(
            epoch, correct, len(test_loader.dataset), 100. * acc, auc
        ))
        return acc, auc, nll
    
    # Train the model
    os.makedirs('./results', exist_ok=True)
    loss_list = []
    acc_list = []
    auc_list = []
    nll_list = []
    times = np.zeros(2)
    for epoch in range(1, num_epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            beginning=timeit.default_timer()
            loss_list.append(train(epoch))
            times[0] += timeit.default_timer()-beginning
            beginning=timeit.default_timer()
            acc, auc, nll = test(epoch)
            times[1] += timeit.default_timer()-beginning
            acc_list.append(acc); auc_list.append(auc); nll_list.append(nll)
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results','qep_'+args.dataset_name+'_checkpoint.dat'))
    
    # save to file
    stats = np.array([acc_list[-1], auc_list[-1], nll_list[-1], times.sum()])
    stats = np.array(['QEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['Method', 'ACC', 'STD', 'NLL', 'time']
    np.savetxt(os.path.join('./results',args.dataset_name+'_QEP.txt'),stats,fmt="%s",delimiter=',',header=','.join(header))
    np.savez_compressed(os.path.join('./results',args.dataset_name+'_QEP'), loss=np.stack(loss_list), acc=np.stack(acc_list), auc=np.stack(auc_list), nll=np.stack(nll_list), times=times)
    
    # plot the result
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    axes[0].plot(loss_list)
    axes[0].set_ylabel('Negative ELBO loss')
    axes[1].plot(acc_list)
    axes[1].set_ylabel('Accuracy')
    axes[2].plot(auc_list)
    axes[2].set_ylabel('AUC')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(os.path.join('./results',args.dataset_name+'_QEP.png'), bbox_inches='tight')
    
if __name__ == '__main__':
    main()