"Deep Q-Exponential Process Classification Model"

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
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialDirichletClassificationLikelihood

from dataset import dataset
os.makedirs('./results', exist_ok=True)

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='mnist')
    parser.add_argument('batch_size', nargs='?', type=int, default=256)
    parser.add_argument('seed', nargs='?', type=int, default=2024)
    args = parser.parse_args()
    # Setting manual seed for reproducibility
    seed = args.seed
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
    
    # Here's a simple standard layer
    class DQEPLayer(DeepQEPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
            self.power = POWER
            inducing_points = torch.randn(output_dims, num_inducing, input_dims, device=device)
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
            # self.covar_module = ScaleKernel(
            #     RBFKernel(
            #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #             np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp
            #         )
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
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateQExponential(mean_x, covar_x, power=self.power)
    
    # define the main model
    class clsDeepQEP(DeepQEP):
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
    
    # define likelihood and model
    hidden_features = [5]
    likelihood = MultitaskQExponentialDirichletClassificationLikelihood(torch.tensor(getattr(train_loader.dataset,{'mnist':'train_labels','cifar10':'targets'}[args.dataset_name])), learn_additional_noise=True)
    likelihood.has_global_noise = True
    model = clsDeepQEP(in_features=input_dim, out_features=num_classes, hidden_features=hidden_features, likelihood=likelihood)
    # set device
    model = model.to(device)
    # likelihood = likelihood.to(device)
    
    # "Loss" for QEPs - the marginal log likelihood
    # mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=len(train_loader.dataset)))
    
    # Use the adam optimizer
    # optimizer = torch.optim.Adam([{'params':model.parameters()},
    #                               {'params':likelihood.parameters()}], lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # lr = 0.001
    # optimizer = torch.optim.SGD([
    #     {'params': model.hyperparameters(), 'lr': lr * 0.1},
    #     {'params': model.variational_parameters()},
    #     {'params': likelihood.parameters()},
    # ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    num_epochs = 1000
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)
    
    # define training and testing procedures
    def train(epoch):
        model.train()
        # likelihood.train()
    
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data.reshape((data.shape[0],-1)))
            loss = -mll(output, likelihood._prepare_targets(target,num_classes=model.likelihood.num_classes)[1]).sum()
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        return loss.item()
    
    def test(epoch):
        model.eval()
        # likelihood.eval()
    
        correct = 0
        lls, aucs = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                appx_dist = model(data.reshape((data.shape[0],-1)))
                pred_mean = appx_dist.mean.mean(0)
                col_max, pred = torch.max(pred_mean,-1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                trans_target = model.likelihood._prepare_targets(target, num_classes=model.likelihood.num_classes)[1]
                ll = model.likelihood.log_marginal(trans_target, appx_dist).mean(0).cpu()
                if model.likelihood.miu:  ll = ll.reshape(1)
                lls.append(ll)
                y_score = torch.exp(pred_mean[:,torch.unique(target)] - col_max[:,None])
                y_score /= y_score.sum(1,keepdims=True)
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
    f_name = args.dataset_name+'_DQEP_dirichlet_'+str(model.num_layers)+'layers_seedNO'+str(seed)
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
        # scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results',f_name+'_checkpoint.dat'))
    
    # save to file
    stats = np.array([acc_list[-1], auc_list[-1], nll_list[-1], times.sum()])
    stats = np.array(['DQEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['Method', 'ACC', 'AUC', 'NLL', 'time']
    np.savetxt(os.path.join('./results',f_name+'.txt'),stats,fmt="%s",delimiter=',',header=','.join(header))
    np.savez_compressed(os.path.join('./results',f_name), loss=np.stack(loss_list), acc=np.stack(acc_list), auc=np.stack(auc_list), nll=np.stack(nll_list), times=times)
    
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
    plt.savefig(os.path.join('./results',f_name+'.png'), bbox_inches='tight')
    
if __name__ == '__main__':
    main()