import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os

# 20 discriminative colors
colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',
          '#469990', '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9']
# celebA qss
classes = ['M+S+Bl','M+S+G','M+S+Br','M+N+Bl','M+N+G','M+N+Br','F+S+Bl','F+S+G','F+S+Br','F+N+Bl','F+N+G','F+N+Br']
# fmnist
filter_labs = []
# classes = ['T+Ts','T+P','T+C','B+Tr','B+D']
# shapenet category
# classes = ['S+Lo','S+C','S+Lc', 'T+W','T+B','T+Te']
# shapenet pose
# classes = ['L+P1','L+P2','L+P3','L+P4', 'C+P1','C+P2','C+P3','C+P4','W+P1','W+P2','W+P3','W+P4','B+P1','B+P2','B+P3','B+P4']
# car pose drop the right profile
# filter_labs = [4, 10, 16, 22]
# classes = ['M+P1','M+P2','M+P3','M+P4','M+P6','SU+P1','SU+P2','SU+P3','SU+P4','SU+P6','Sp+P1','Sp+P2','Sp+P3','Sp+P4',
#            'Sp+P6','Se+P1','Se+P2','Se+P3','Se+P4', 'Se+P6']
# imagenet category
# classes = ['H+E','H+P','H+Si', 'H+Ta','D+Cor','D+G', 'D+Hu', 'D+Sa', 'B+Cou', 'B+Le', 'B+Li', 'B+Ti']

# RAF
# classes = ['M+S', 'M+N', 'F+S', 'F+N']

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, required=True, help="feature file save path")

opts = parser.parse_args()
file_path = opts.file_path
fea_lab = np.load(file_path)
folder_path = file_path[0:file_path.rfind('/')]

feac = fea_lab['content']
num = np.shape(feac)[0]
feac = np.reshape(feac, [num, -1])
fea1 = fea_lab['h2_fea']
fea1 = np.reshape(fea1, [num, -1])
fea2 = fea_lab['h1_fea']
fea2 = np.reshape(fea2, [num, -1])
fea12 = np.concatenate((fea1,fea2), 1)
fea3 = fea_lab['h0_fea']
fea3 = np.reshape(fea3, [num, -1])
fea23 = np.concatenate((fea2, fea3), 1)
fea123 = np.concatenate((fea1,fea2,fea3), 1)

lab = fea_lab['h0_lab']
lab_list = np.unique(lab)

tsne = TSNE(n_components=2)

# qss
X_tsne = tsne.fit_transform(fea23)
level = 'fea23'

plt.figure(figsize=(12, 6))
count = 0
for c_id in lab_list:
    if c_id in filter_labs:
        continue
    idx = np.argwhere(lab == c_id)
    tmp_tsne = X_tsne[idx].squeeze()
    plt.scatter(x=tmp_tsne[:,0], y=tmp_tsne[:,1], alpha=1, color=colors[count], label= classes[count])
    count = count + 1
plt.legend()
plt.savefig(os.path.join(folder_path, 'tsne_'+level+'.pdf'))
plt.show()
plt.close()