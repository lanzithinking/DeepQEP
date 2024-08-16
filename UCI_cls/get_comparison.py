"Plot comparison of various models"

import os
import numpy as np

os.makedirs('./results', exist_ok=True)

datasets=('car', 'haberman', 'nursery', 'tic_tac_toe', 'website_phishing', 'seismic')
num_datasets=len(datasets)
algs=('DGP', 'DQEP', 'DKLGP', 'DKLQEP', 'DSPP')
alg_names=('DeepGP', 'DeepQEP', 'DKLGP', 'DKLQEP', 'DSPP')
num_algs=len(algs)

# set the format
folder = './results'
header = ('Dataset', 'Method', 'ACC', 'AUC', 'NLL', 'time')
incl_header = True
fmt = ('%s',)*2+('%.4f',)*4
# obtain results
txt_files=[f for f in os.listdir(folder) if f.endswith('.txt')]
for i in range(num_datasets):
    for j in range(num_algs):
        formats = ('i4','S'+str(len(alg_names[j])))+('f4',)*5
        for f_i in txt_files:
            if datasets[i]+'_'+algs[j] in f_i:
                f_read=np.loadtxt(os.path.join(folder,f_i), delimiter=',', dtype={'names':header,'formats':formats})
                stats = np.stack([list(r_i)[2:] for r_i in f_read])
                stats = stats[np.isfinite(stats).all(axis=1)]
                with open(os.path.join(folder, 'summary_mean.csv'),'ab') as f:
                    np.savetxt(f, np.concatenate([[datasets[i],alg_names[j]],stats.mean(0)])[None,:], fmt='%s', delimiter=',', header=','.join(header) if incl_header else '')
                f.close()
                with open(os.path.join(folder, 'summary_std.csv'),'ab') as f:
                    np.savetxt(f, np.concatenate([[datasets[i],alg_names[j]],stats.std(0,ddof=1)])[None,:], fmt='%s', delimiter=',', header=','.join(header) if incl_header else '')
                f.close()
                incl_header = False
