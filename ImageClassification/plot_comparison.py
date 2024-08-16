"Plot comparison of various models"

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('./results', exist_ok=True)

dataset_name={0:'mnist',1:'cifar10'}[1]
algs=('CNN', 'DKLGP', 'DKLQEP')
alg_names=('CNN', 'DKL-GP', 'DKL-QEP')
num_algs=len(algs)

# plot the results
fig, axes = plt.subplots(1,2, figsize=(12,5))
lty = ('-', '--', '-.')

# obtain results
folder = './results'
npz_files=[f for f in os.listdir(folder) if f.endswith('.npz')]
for i in range(num_algs):
    for f_i in npz_files:
        if dataset_name+'_'+algs[i] in f_i:
            f_read=np.load(os.path.join(folder,f_i))
            loss, acc = f_read['loss'], f_read['acc']
            axes[0].plot(loss, lty[i]); axes[0].set_yscale('log')
            axes[0].set_xlabel('iteration', fontsize=20)
            axes[0].set_ylabel('Negative ELBO loss', fontsize=20)
            axes[1].plot(acc, lty[i]); #axes[1].set_ylim(.85,1)
            axes[1].set_xlabel('iteration', fontsize=20)
            axes[1].set_ylabel('Accuracy', fontsize=20)
axes[0].legend(alg_names, fontsize=18)
axes[1].legend(alg_names, fontsize=18)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=14)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig(os.path.join('./results',dataset_name+'.png'), bbox_inches='tight')

