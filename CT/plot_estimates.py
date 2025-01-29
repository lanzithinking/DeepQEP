"""
Plot estimates of uncertainty field u in linear inverse problem of Shepp-Logan head phantom.
----------------------
Shiwei Lan @ ASU, 2025
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp


# models
mdls=('DGP_6layers','DQEP_6layers','DSPP_6layers')
mdl_names=['Deep GP','Deep QEP','DSPP']
num_mdls=len(mdls)
# obtain estimates
folder = './results'
if os.path.exists(os.path.join(folder,'summary.pckl')):
    f=open(os.path.join(folder,'summary.pckl'),'rb')
    truth, opt_X=pickle.load(f)
    f.close()
    print('summary.pckl has been read!')
else:
    opt_X=[[]]*num_mdls
    for m in range(num_mdls):
        # preparation for estimates
        pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
        for f_i in pckl_files:
            if '_'+mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    truth=f_read[0]
                    opt_X[m]=f_read[1]
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'summary.pckl'),'wb')
    pickle.dump([truth,opt_X],f)
    f.close()

# plot 
plt.rcParams['image.cmap'] = 'gray'
num_rows=1
titles = ['Truth']+mdl_names

# estimates
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=truth if i==0 else opt_X[i-1]
    plt.imshow(img,extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/estimates.png',bbox_inches='tight')
# plt.show()
