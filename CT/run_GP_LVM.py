"Gaussian Latent Variable Model"

import os,argparse,pickle
import random
import numpy as np
import scipy.sparse.linalg as spsla
import timeit
# from skimage.metrics import structural_similarity as ssim_f
from haar_psi import haar_psi_numpy
import matplotlib.pylab as plt

import torch
import tqdm
# from tqdm.notebook import trange

# gpytorch imports
import sys
sys.path.insert(0,'../GPyTorch')
import gpytorch
from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.means import ZeroMean, ConstantMean, LinearMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LinearKernel, RFFKernel
from gpytorch.distributions import MultivariateNormal

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
    
    # define model
    class bGPLVM(BayesianGPLVM):
        def __init__(self, n, data_dim, latent_dim, n_inducing, lsqr=False):
            self.n = n
            self.batch_shape = torch.Size([data_dim])
    
            # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
            # regularly placed with shape (D x n_inducing x latent_dim).
            self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)
    
            # Sparse Variational Formulation (inducing variables initialised as randn)
            q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
            q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
    
            # Define prior for X
            X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
            prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean)*100)
    
            # Initialise X c
            if lsqr == True:
                X_init = torch.nn.Parameter(torch.tensor(spsla.lsqr(A=proj, b=sino, damp=0.1)[0], dtype=torch.float32)+1e-4*torch.rand(n, latent_dim)) # Initialise X to least square solution
                # X_init = torch.nn.Parameter(torch.linalg.lstsq(projector, sinogram)[0])
            else:
                # X_init = torch.nn.Parameter(torch.randn(latent_dim))
                X_init = torch.nn.Parameter(torch.randn(n, latent_dim))
    
            # LatentVariable (c)
            X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
    
            # For (a) or (b) change to below:
            # X = PointLatentVariable(n, latent_dim, X_init)
            # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
    
            super().__init__(X, q_f)
    
            # # Kernel (acting on latent dimensions)
            # self.mean_module = ConstantMean(ard_num_dims=latent_dim)
            # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
            # Kernel (acting on transformed latent dimensions)
            self.mean_module = ConstantMean(ard_num_dims=n)#data_dim)
            # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n))#data_dim))
            # self.covar_module = ScaleKernel(LinearKernel(ard_num_dims=n))
            # self.covar_module = ScaleKernel(
            #     MaternKernel(nu=1.5, batch_shape=self.batch_shape, ard_num_dims=n,
            #         # lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
            #         #     np.exp(-1), np.exp(1), sigma=0.1, transform=torch.exp)
            #     ),
            #     batch_shape=self.batch_shape,
            # )
            self.covar_module = ScaleKernel(
                RFFKernel(num_samples=data_dim, num_dims=n),
                batch_shape=self.batch_shape,
            )
    
        def forward(self, X):
            proj_X = torch.bmm(X, projector)#.permute((1,2,0)))#.mean(0)
            # proj_X = torch.tensordot(X, projector, ((0,2),(1,2)))
            mean_ = self.mean_module(proj_X)
            covar_ = self.covar_module(proj_X)
            dist = MultivariateNormal(mean_, covar_)
            return dist
    
        def _get_batch_idx(self, batch_size):
            valid_indices = np.arange(self.n)
            batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
            return np.sort(batch_indices)
    
    
    # N = len(sinogram)
    # data_dim = sinogram.shape[1]
    N, data_dim = sinogram.shape
    # latent_dim = projector.shape[-1]
    latent_dim = projector.shape[1]
    n_inducing = 200
    lsqr = True
    
    # Model
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, lsqr=lsqr)
    
    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    
    # Declaring the objective to be optimised along with optimiser
    # (see models/latent_variable.py for how the additional loss terms are accounted for)
    mll = VariationalELBO(likelihood, model, num_data=N, beta=100)
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.1)
    
    # set device
    projector = projector.to(device)
    sinogram = sinogram.to(device)
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
    # using the optimizer provided.
    
    loss_list = []
    num_epochs = 10000
    # iterator = trange(num_epochs, leave=True)
    iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
    beginning=timeit.default_timer()
    for i in iterator:
        optimizer.zero_grad()
        sample = model.sample_latent_variable()  # a full sample returns latent x across all N
        if sample.ndim==1: sample = sample.unsqueeze(0)
        output = model(sample)
        loss = -mll(output, sinogram.T).sum()
        # iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        print('Epoch {}/{}: Loss: {}'.format(i, num_epochs, loss.item() ))
    time_ = timeit.default_timer()-beginning
    print('Training uses: {} seconds.'.format(time_))
    
    # obtain estimates
    X = (model.X.q_mu if hasattr(model.X, 'q_mu') else model.X.X).detach().cpu().numpy().mean(0).reshape(truth.shape,order='F')
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
    filename = 'CT_proj'+str(args.n_angles)+'_GP-LVM'
    f=open(os.path.join('./results',filename+'_seed'+str(seed)+'.pckl'),'wb')
    pickle.dump([truth, X, loss_list],f)
    f.close()
    # stats = np.array([rem, psnr, ssim, haarpsi, time_])
    stats = np.array([rem, psnr, haarpsi, time_])
    stats = np.array([seed,'GP-LVM']+[np.array2string(r, precision=4) for r in stats])[None,:]
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