"Deep Gaussian Process Classification Model"

import os, argparse
import math
import random
import numpy as np
import timeit

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
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskDirichletClassificationLikelihood

from dataset import dataset
os.makedirs('./results', exist_ok=True)

def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', nargs='?', type=str, default='cifar10')
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
    
    # load data
    train_loader, test_loader, feature_extractor, num_classes = dataset(args.dataset_name, seed, batch_size=args.batch_size)
    # input_dim = np.prod(list(train_loader.dataset.data.shape[1:]))
    input_dim = np.prod(next(iter(train_loader))[0].shape[1:])
    
    # Here's a simple standard layer
    class DGPLayer(DeepGPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
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
    
            super().__init__(variational_strategy, input_dims, output_dims)
            self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(input_dims)}[mean_type]
            # self.covar_module = ScaleKernel(
            #     RBFKernel(
            #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            #         )
            #     )
            # )
            self.covar_module = ScaleKernel(
                MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=input_dims,
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)
                ),
                batch_shape=batch_shape,
            )
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)
    
    # define the main model
    hidden_features = [5]
    
    class clsDeepGP(DeepGP):
        def __init__(self, in_features, out_features, hidden_features=2):
            super().__init__()
            if isinstance(hidden_features, int):
                layer_config = torch.cat([torch.arange(in_features, out_features, step=(out_features-in_features)/max(1,hidden_features)).type(torch.int), torch.tensor([out_features])])
            elif isinstance(hidden_features, list):
                layer_config = [in_features]+hidden_features+[out_features]
            layers = []
            for i in range(len(layer_config)-1):
                layers.append(DGPLayer(
                    input_dims=layer_config[i],
                    output_dims=layer_config[i+1],
                    mean_type='linear' if i < len(layer_config)-2 else 'constant'
                ))
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
    
        def forward(self, inputs):
            output = self.layers[0](inputs)
            for i in range(1,len(self.layers)):
                output = self.layers[i](output)
            return output
    
    # define likelihood and model
    likelihood = MultitaskDirichletClassificationLikelihood(torch.tensor(getattr(train_loader.dataset,{'mnist':'train_labels','cifar10':'targets'}[args.dataset_name])), learn_additional_noise=True)
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
    model = clsDeepGP(in_features=input_dim, out_features=num_classes, hidden_features=hidden_features)
    # set device
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # "Loss" for GPs - the marginal log likelihood
    # mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))
    mll = DeepApproximateMLL(VariationalELBO(likelihood, model, num_data=len(train_loader.dataset)))
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params':model.parameters()},
                                  {'params':likelihood.parameters()}], lr=0.001)
    # lr = 0.001
    # optimizer = torch.optim.SGD([
    #     {'params': model.hyperparameters(), 'lr': lr * 0.1},
    #     {'params': model.variational_parameters()},
    #     {'params': likelihood.parameters()},
    # ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    num_epochs = 2000
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * num_epochs, 0.75 * num_epochs], gamma=0.1)
    
    # define training and testing procedures
    def train(epoch):
        model.train()
        likelihood.train()
    
        minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data.flatten(1))
            loss = -mll(output, likelihood._prepare_targets(target,num_classes=likelihood.num_classes)[1]).sum()
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        return loss.item()
    
    def test(epoch):
        model.eval()
        likelihood.eval()
    
        correct = 0
        vars = []
        lls = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                appx_dist = model(data.flatten(1))
                pred = appx_dist.mean.mean(0).argmax(-1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                vars.append(appx_dist.variance.mean(0).median(-1)[0].cpu())
                lls.append(likelihood.log_marginal(target[:,None], appx_dist).mean(0).cpu())
        acc = correct / float(len(test_loader.dataset))
        print('Epoch {}: test set: Accuracy: {}/{} ({}%)'.format(
            epoch, correct, len(test_loader.dataset), 100. * acc
        ))
        return acc.item(), torch.cat(vars, 0).sqrt().mean().item(), -torch.cat(lls, 0).mean().item()
    
    # Train the model
    os.makedirs('./results', exist_ok=True)
    loss_list = []
    acc_list = []
    std_list = []
    nll_list = []
    times = np.zeros(2)
    for epoch in range(1, num_epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            beginning=timeit.default_timer()
            loss_list.append(train(epoch))
            times[0] += timeit.default_timer()-beginning
            beginning=timeit.default_timer()
            acc, std, nll = test(epoch)
            times[1] += timeit.default_timer()-beginning
            acc_list.append(acc); std_list.append(std); nll_list.append(nll)
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, os.path.join('./results','dgp_'+str(model.num_layers)+'layers_'+args.dataset_name+'_checkpoint.dat'))
    
    # save to file
    stats = np.array([acc_list[-1], std_list[-1], nll_list[-1], times.sum()])
    stats = np.array(['DGP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['Method', 'ACC', 'STD', 'NLL', 'time']
    np.savetxt(os.path.join('./results',args.dataset_name+'_DGP_'+str(model.num_layers)+'layers.txt'),stats,fmt="%s",delimiter=',',header=','.join(header))
    np.savez_compressed(os.path.join('./results',args.dataset_name+'_DGP_'+str(model.num_layers)+'layers'), loss=np.stack(loss_list), acc=np.stack(acc_list), std=np.stack(std_list), nll=np.stack(nll_list), times=times)
    
    # plot the result
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    axes[0].plot(loss_list)
    axes[0].set_ylabel('Negative ELBO loss')
    axes[1].plot(acc_list)
    axes[1].set_ylabel('Accuracy')
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig(os.path.join('./results',args.dataset_name+'_DGP_'+str(model.num_layers)+'layers.png'), bbox_inches='tight')

if __name__ == '__main__':
    main()