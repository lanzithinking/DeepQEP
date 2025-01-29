"Deep Sigma Point Process Model"

import os,argparse,pickle
import random
import numpy as np
import scipy.sparse.linalg as spsla
import timeit
# from skimage.metrics import structural_similarity as ssim_f
from haar_psi import haar_psi_numpy
from matplotlib import pyplot as plt

import torch
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, LinearKernel, RFFKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.deep_gps.dspp import DSPPLayer, DSPP
from gpytorch.mlls import DeepPredictiveLogLikelihood
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# image reconstruction metrics
def PSNR(reco, gt):
    mse = np.mean((np.asarray(reco) - gt)**2)
    if mse == 0.:
        return float('inf')
    data_range = (np.max(gt) - np.min(gt))
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reco, gt):
    data_range = (np.max(gt) - np.min(gt))
    return ssim_f(reco, gt, data_range=data_range)


def main(seed=2024):
    parser = argparse.ArgumentParser()
    parser.add_argument('n_angles', nargs='?', type=int, default=90)
    args = parser.parse_args()
    
    # load CT data
    loaded=np.load(os.path.join('./','CT_obs_proj'+str(args.n_angles)+'.npz'),allow_pickle=True)
    proj=loaded['proj'][0]
    projector=torch.tensor(proj.toarray().reshape((args.n_angles, -1, proj.shape[-1]),order='F'), dtype=torch.float32)
    sino=loaded['obs']
    sinogram=torch.tensor(sino.reshape((args.n_angles,-1),order='F'), dtype=torch.float32)
    nzvar=torch.tensor(loaded['nzvar']); truth=loaded['truth']
    # permute projector
    projector = projector.permute((1,2,0))
    
    # Setting manual seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using the '+device+' device...')
    
    # Here's a simple standard layer
    class DSPP_Layer(DSPPLayer):
        def __init__(self, input_dims, output_dims, num_inducing=200, mean_type='constant', Q=8, **kwargs):
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
            n = kwargs.pop('n',1); 
            # self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
            self.mean_module = {'constant': ConstantMean(), 'linear': LinearMean(n)}[mean_type]
            # self.covar_module = ScaleKernel(
            #     RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims,
            #         lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
            #         )
            #     )
            # )
            # self.covar_module = ScaleKernel(
            #     MaternKernel(nu=1.5, batch_shape=batch_shape, ard_num_dims=n,
            #         # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #         #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
            #     ),
            #     batch_shape=batch_shape,
            # )
            # self.covar_module = ScaleKernel(LinearKernel(ard_num_dims=n))
            self.covar_module = ScaleKernel(
                RFFKernel(num_samples=output_dims, num_dims=n),
                batch_shape=batch_shape,
            )
            
            # LatentVariable (c)
            data_dim = output_dims; latent_dim = input_dims
            latent_prior_mean = torch.zeros(n, latent_dim)
            latent_prior = NormalPrior(latent_prior_mean, torch.ones_like(latent_prior_mean)*100)
            latent_init = kwargs.pop('latent_init', None)
            if latent_init is not None:
                self.latent_variable = VariationalLatentVariable(n, data_dim, latent_dim, torch.nn.Parameter(latent_init), latent_prior)
    
            # For (a) or (b) change to below:
            # X = PointLatentVariable(n, latent_dim, latent_init)
            # X = MAPLatentVariable(n, latent_dim, latent_init, latent_prior)
    
        def forward(self, x, projection=None):
            x_ = x if projection is None else torch.bmm(x, projector)#.permute((1,2,0)))
            mean_x = self.mean_module(x_)
            covar_x = self.covar_module(x_)
            return MultivariateNormal(mean_x, covar_x)
    
    # define the main model
    class MultitaskDSPP(DSPP):
        def __init__(self, n, in_features, out_features, hidden_features=2, latent_init=None):
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
                    n=n if i==0 else layer_config[i],
                    latent_init=latent_init if i==0 else None
                ))
            self.num_layers = len(layers)
            self.layers = torch.nn.Sequential(*layers)
    
            # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
            self.likelihood = MultitaskGaussianLikelihood(num_tasks=out_features)
    
        def forward(self, inputs):
            output = self.layers[0](inputs, projection=projector)#, are_samples=True)
            for i in range(1,len(self.layers)):
                output = self.layers[i](output)
            return output
    
    n = sinogram.shape[0]
    # latent_dim = projector.shape[-1]
    latent_dim = projector.shape[1]
    num_tasks = sinogram.size(-1)
    hidden_features = [num_tasks]+[10]*4
    num_quadrature_sites = 8
    lsqr = True
    latent_init=torch.tensor(spsla.lsqr(A=proj, b=sino, damp=0.1)[0], dtype=torch.float32)+1e-4*torch.rand(n, latent_dim) if lsqr else torch.randn(n, latent_dim)
    model = MultitaskDSPP(n, latent_dim, num_tasks, hidden_features, latent_init)
    
    # training
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = DeepPredictiveLogLikelihood(model.likelihood, model, num_data=n, beta=100)
    
    # set device
    projector = projector.to(device)
    sinogram = sinogram.to(device)
    model = model.to(device)
    mll = mll.to(device)
    
    loss_list = []
    num_epochs = 10000
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning=timeit.default_timer()
    for i in iterator:
        optimizer.zero_grad()
        sample = model.layers[0].latent_variable()
        if sample.ndim==1: sample = sample.unsqueeze(0)
        output = model(sample)
        loss = -mll(output, sinogram).sum()
        # iterator.set_postfix(loss=loss.item())
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
        print('Epoch {}/{}: Loss: {}'.format(i, num_epochs, loss.item() ))
    time_ = timeit.default_timer()-beginning
    print('Training uses: {} seconds.'.format(time_))
    
    # load the best model
    model.load_state_dict(optim_model)
    model.eval()
    
    # obtain estimates
    X = getattr(model.layers[0].latent_variable, 'q_mu' if isinstance(model.layers[0].latent_variable, VariationalLatentVariable) else 'X')
    X = X.detach().cpu().numpy().mean(0).reshape(truth.shape,order='F')
    rem = np.linalg.norm(X-truth)/np.linalg.norm(truth)
    psnr = PSNR(X, truth)
    # ssim = SSIM(X, truth)
    X_ = ((X - np.min(X)) * (255.0/(np.max(X) - np.min(X)))) #.astype('uint8')
    haarpsi = haar_psi_numpy(X_,truth)[0]
    print('REM: {}'.format(rem))
    print('PSNR: {}'.format(psnr))
    # print('SSIM: {}'.format(ssim))
    print('HaarPSI: {}'.format(haarpsi))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    filename = 'CT_proj'+str(args.n_angles)+'_DSPP_'+str(model.num_layers)+'layers'
    f=open(os.path.join('./results',filename+'_seed'+str(seed)+'.pckl'),'wb')
    pickle.dump([truth, X, loss_list],f)
    f.close()
    # stats = np.array([rem, psnr, ssim, haarpsi, time_])
    stats = np.array([rem, psnr, haarpsi, time_])
    stats = np.array([seed,'DSPP']+[np.array2string(r, precision=4) for r in stats])[None,:]
    # header = ['seed', 'Method', 'REM', 'PSNR', 'SSIM', 'HaarPSI', 'time']
    header = ['seed', 'Method', 'REM', 'PSNR', 'HaarPSI', 'time']
    with open(os.path.join('./results',filename+'.txt'),'ab') as f_:
        np.savetxt(f_,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')
    
    # # plot results
    # plt.figure(figsize=(20, 7))
    # plt.set_cmap('gray')#'Greys')
    # plt.subplot(131)
    # plt.imshow(truth, extent=[0, 1, 0, 1])
    # plt.title('Truth')
    # plt.subplot(132)
    # plt.imshow(X, extent=[0, 1, 0, 1])
    # plt.title('Solution')
    # plt.subplot(133)
    # plt.plot(loss_list)
    # plt.title('Neg. ELBO Loss')
    # # plt.show()
    # # os.makedirs('./results', exist_ok=True)
    # plt.savefig(os.path.join('./results',filename+'.png'),bbox_inches='tight')

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