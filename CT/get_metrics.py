"""
Get various metrics of estimates for uncertainty field u in linear inverse problem of Shepp-Logan head phantom.
----------------------
Shiwei Lan @ ASU, 2025
"""

import os,pickle
import numpy as np
from skimage.metrics import structural_similarity as ssim_f
from haar_psi import haar_psi_numpy
import itertools

def PSNR(reco, gt):
    mse = np.mean((np.asarray(reco) - gt)**2)
    if mse == 0.:
        return float('inf')
    data_range = (np.max(gt) - np.min(gt))
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reco, gt):
    data_range = (np.max(gt) - np.min(gt))
    return ssim_f(reco, gt, data_range=data_range)

# models
mdls=['GP-LVM','QEP-LVM']+[i+'_'+str(j)+'layers' for i,j in itertools.product(('DGP','DQEP','DSPP'), range(2,7))]
# mdls=('DGP_6layers','DQEP_6layers','DSPP_6layers')
mdl_names=['GP-LVM','QEP-LVM']+[i+'-'+str(j) for i,j in itertools.product(('Deep GP','Deep QEP','DSPP'), range(2,7))]
num_mdls=len(mdls)
# store results
rle_m=np.zeros(num_mdls); rle_s=np.zeros(num_mdls)
psnr_m=np.zeros(num_mdls); psnr_s=np.zeros(num_mdls)
ssim_m=np.zeros(num_mdls); ssim_s=np.zeros(num_mdls)
haarpsi_m=np.zeros(num_mdls); haarpsi_s=np.zeros(num_mdls)
# obtain estimates
folder = './results'
if not os.path.exists(os.path.join(folder,'summary.pckl')):
    opt_X=[[]]*num_mdls
for m in range(num_mdls):
    print('Processing '+mdls[m]+' model...\n')
    # preparation for estimates
    errs=[]; psnr=[]; ssim=[]; haarpsi=[]; files_read=[]
    num_read=0
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    for f_i in pckl_files:
        if '_'+mdls[m]+'_' in f_i:
            try:
                f=open(os.path.join(folder,f_i),'rb')
                f_read=pickle.load(f)
                truth=f_read[0]
                X=f_read[1]
                # compute error
                # errs.append(np.linalg.norm(X-truth)/np.linalg.norm(truth))
                errs.append(np.linalg.norm(X[:]-truth[:],1)/np.linalg.norm(truth[:],1))
                psnr.append(PSNR(X, truth))
                ssim.append(SSIM(X, truth))
                X_ = ((X - np.min(X)) * (255.0/(np.max(X) - np.min(X)))) #.astype('uint8')
                haarpsi.append(haar_psi_numpy(X_,truth)[0])
                files_read.append(f_i)
                num_read+=1
                f.close()
                print(f_i+' has been read!')
            except:
                pass
    print('%d experiment(s) have been processed for %s model.' % (num_read, mdls[m]))
    if num_read>0:
        errs = np.stack(errs)
        rle_m[m] = np.median(errs)
        rle_s[m] = errs.std()
        psnr_m[m] = np.median(psnr)
        psnr_s[m] = np.std(psnr)
        ssim_m[m] = np.median(ssim)
        ssim_s[m] = np.std(ssim)
        haarpsi_m[m] = np.median(haarpsi)
        haarpsi_s[m] = np.std(haarpsi)
        # get the best for plotting
        if not os.path.exists(os.path.join(folder,'summary.pckl')):
            f_i=files_read[np.argmin(errs)]
            f=open(os.path.join(folder,f_i),'rb')
            f_read=pickle.load(f)
            opt_X[m]=f_read[1]
            f.close()
            print(f_i+' has been selected for plotting.')
if not os.path.exists(os.path.join(folder,'summary.pckl')):
    f=open(os.path.join(folder,'summary.pckl'),'wb')
    pickle.dump([truth,opt_X],f)
    f.close()

# save
import pandas as pd
means = pd.DataFrame(data=np.vstack((rle_m,psnr_m,ssim_m,haarpsi_m)),columns=mdl_names[:num_mdls],index=['rle','psnr','ssim','haarpsi'])
stds = pd.DataFrame(data=np.vstack((rle_s,psnr_s,ssim_s,haarpsi_s)),columns=mdl_names[:num_mdls],index=['rle','psnr','ssim','haarpsi'])
# means.to_csv(os.path.join(folder,'MET-mean.csv'),columns=mdl_names[:num_mdls])
# stds.to_csv(os.path.join(folder,'MET-std.csv'),columns=mdl_names[:num_mdls])

means = means.drop('psnr')
stds = stds.drop('psnr')
import matplotlib.pyplot as plt
# plot
# fig,axes = plt.subplots(nrows=1,ncols=len(means),sharex=True,sharey=False,figsize=(21,4))
fig,axes = plt.subplots(nrows=1,ncols=len(means),sharex=True,sharey=False,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    for j in ('GP','QEP','DSPP'):
        col_j = [col for col in means.columns if j in col]
        m_i = means.iloc[i].loc[col_j]
        s_i = stds.iloc[i].loc[col_j]
        if j=='DSPP':
            m_i = pd.concat([pd.Series([float('NaN')]),m_i])
            s_i = pd.concat([pd.Series([float('NaN')]),s_i])
        ax.errorbar(range(len(m_i)),m_i, yerr=s_i, label=j)
        ax.set_xticks(range(len(m_i)),range(1,1+len(m_i)))
        ax.set_xlabel('layers', fontsize=18)
        plt.legend(['Deep GP', 'Deep QEP', 'DSPP'],fontsize=16, frameon=False)
    # ax.set_ylabel(means.index[i], fontsize=18)
    # ax.set_ylabel({0:'Relative Error',1:'PSNR',2:'SSIM',3:'HaarPSI'}[i], fontsize=18)
    ax.set_ylabel({0:'Relative Error',1:'SSIM',2:'HaarPSI'}[i], fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/metrics.png',bbox_inches='tight')
# plt.show()