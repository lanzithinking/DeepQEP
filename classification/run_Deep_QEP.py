"Deep Q-Exponential Process Classification Model"

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
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateQExponential
from gpytorch.models.deep_qeps import DeepQEPLayer, DeepQEP, DeepLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskQExponentialDirichletClassificationLikelihood

POWER = 1.0

# generate data
dataset = {0:'rhombus',1:'annulus'}[0]
def gen_data(num_data, seed = 2019):
    torch.random.manual_seed(seed)

    x = torch.randn(num_data,1)
    y = torch.randn(num_data,1)

    u = torch.rand(1)
    # data_fn = lambda x, y: 1 * torch.sin(0.15 * u * 3.1415 * (x + y)) + 1
    data_fn = lambda x, y: {'annulus': 1 * torch.cos(0.4 * u * np.pi * np.sqrt(x**2 + y**2)) + 1,
                            'rhombus': 1 * torch.cos(0.4 * u * np.pi * np.abs(x) + np.abs(y)) + 1}[dataset]
    latent_fn = data_fn(x, y)
    z = torch.round(latent_fn).long().squeeze()
    return torch.cat((x,y),dim=1), z, data_fn
n_train = 500
train_x, train_y, genfn = gen_data(n_train)

# testing data
n_test = 50
test_d1 = np.linspace(-3, 3, n_test)
test_d2 = np.linspace(-3, 3, n_test)

test_x_mat, test_y_mat = np.meshgrid(test_d1, test_d2)
test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)

test_x = torch.cat((test_x_mat.view(-1,1), test_y_mat.view(-1,1)),dim=1)
test_labels = torch.round(genfn(test_x_mat, test_y_mat))
test_y = test_labels.long().view(-1)

# # plot data with true boundary
# sys.path.append('../')
# from util.gpmkr_scatter import gpmkr_scatter
# os.makedirs('./results_'+dataset, exist_ok=True)
# mkrs = ['^', 'o', 'v']
# fig = plt.figure(figsize=(5, 5))
# plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
# gpmkr_scatter(train_x[:,0].numpy(), train_x[:,1].numpy(), m = [mkrs[int(i)] for i in train_y], facecolors='none', edgecolors='r')
# plt.title('True Labels', fontsize=20)
# plt.savefig('./results_'+dataset+'/cls_truth.png',bbox_inches='tight')
# plt.show()

def main(seed=2024):
    
    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Here's a simple standard layer
    class DQEPLayer(DeepQEPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=64, mean_type='constant'):
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
            # self.covar_module = ScaleKernel(
            #     RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            #     batch_shape=batch_shape,
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
    
    hidden_features = [3]
    # we let the DirichletClassificationLikelihood compute the targets for us
    likelihood = MultitaskQExponentialDirichletClassificationLikelihood(train_y, learn_additional_noise=True, power=torch.tensor(POWER), miu=False)
    likelihood.has_global_noise = True
    model = clsDeepQEP(in_features=train_x.shape[-1], out_features=likelihood.num_classes, hidden_features=hidden_features, likelihood=likelihood)
    
    
    # Find optimal model hyperparameters
    model.train()
    # likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes QExponentialLikelihood parameters
    
    # "Loss" for QEPs - the marginal log likelihood
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
    # mll = DeepLikelihood(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))
    
    loss_list = []
    training_iter = 1000
    beginning=timeit.default_timer()
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, model.likelihood.transformed_targets.T).sum()
        loss.backward()
        if i % 5 == 0:
            print('Iter %d/%d - Loss: %.3f last layer lengthscale: %.3f   noise: %.3f   accuracy: %.4f' % (
                i + 1, training_iter, loss.item(),
                model.layers[-1].covar_module.base_kernel.lengthscale.mean().item(),
                model.likelihood.second_noise_covar.noise.mean().item(),
                output.mean.mean(0).argmax(-1).eq(train_y).mean(dtype=float).item()
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
    
    # load the best model
    model.load_state_dict(optim_model)
    # prediction
    model.eval()
    # likelihood.eval()
    
    from sklearn.metrics import roc_auc_score
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)
        pred_means = test_dist.mean.mean(0)
        col_max, pred = torch.max(pred_means,-1)
        test_targets = model.likelihood._prepare_targets(test_y, num_classes=model.likelihood.num_classes)[1]
        lls = model.likelihood.log_marginal(test_targets, test_dist).mean(0)
        y_score = torch.exp(pred_means - col_max[:,None])
        y_score /= y_score.sum(1,keepdims=True)
    NLL = -lls.mean().item()
    ACC = pred.eq(test_y).mean(dtype=float).item()
    AUC = roc_auc_score(test_y, y_score, multi_class='ovo').item()
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test NLL: {}'.format(NLL))
    
    # save to file
    os.makedirs('./results_'+dataset, exist_ok=True)
    stats = np.array([ACC, AUC, NLL, time_])
    stats = np.array([seed,'DeepQEP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'NLL', 'time']
    f_name = os.path.join('./results_'+dataset,'cls_DeepQEP.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')
    
    # # plots
    # os.makedirs('./results_'+dataset, exist_ok=True)
    #
    # # logits
    # fig, axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(15,5))
    # sub_figs = [None]*len(axes.flat)
    # for i,ax in enumerate(axes.flat):
    #     plt.axes(ax)
    #     sub_figs[i]=plt.contourf(
    #         test_x_mat.numpy(), test_y_mat.numpy(), pred_means[:,i].numpy().reshape((n_test,n_test))
    #     )
    #     ax.set_title("Logits: Class " + str(i), fontsize = 20)
    #     ax.set_aspect('auto')
    #     plt.axis([-3, 3, -3, 3])
    # # set color bar
    # # cax,kw = mp.colorbar.make_axes([ax for ax in axes.flat])
    # # plt.colorbar(sub_fig, cax=cax, **kw)
    # sys.path.append('../')
    # from util.common_colorbar import common_colorbar
    # fig=common_colorbar(fig,axes,sub_figs)
    # plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # plt.savefig('./results_'+dataset+'/cls_logits_DeepQEP_'+str(model.num_layers)+'layers.png',bbox_inches='tight')
    #
    # # boundaries
    # fig = plt.figure(figsize=(5, 5))
    # plt.contourf(test_x_mat.numpy(), test_y_mat.numpy(), pred_means.max(1)[1].reshape((n_test,n_test)))
    # plt.title('Deep QEP', fontsize=20)
    # plt.savefig('./results_'+dataset+'/cls_boundaries_DeepQEP_'+str(model.num_layers)+'layers.png',bbox_inches='tight')
    
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