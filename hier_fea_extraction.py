from __future__ import print_function
from utils import get_config
from trainer import HDN_Trainer
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import utils as utl
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.decomposition import PCA
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--save_name', type=str, required=True, help="feature file save name")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader
trainer = HDN_Trainer(config)

_, test_loader = utl.get_all_data_loaders(config)
# unseen_root = '/home/qiaoshishi/datasets/RAF/'
# unseen_file = 'test_race.txt'
# unseen_tree = [4, 3, 3, 3, 3, 2, 2, 2]
# unseen_filter_label = None
# unseen_new_size_a = 128
# unseen_height = 128
# unseen_width = 128
# unseen_crop = False
# test_loader = get_data_loader_list(unseen_root, unseen_file, unseen_filter_label, unseen_tree, 16, False,
#                                         unseen_new_size_a, unseen_height, unseen_width, 3, unseen_crop, False)

model_name = config['model_name']
output_directory = os.path.join("./results", model_name)


if not os.path.exists(output_directory):
    print("Creating directory: {}".format(output_directory))
    os.makedirs(output_directory)

state_dict = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict['gen'])

trainer.cuda()
trainer.eval()
encode = trainer.gen.encode

# Start testing
count = 0
sample_num = test_loader.dataset.__len__()
max_num = 500
content_fea = []
h0_fea = []
h1_fea = []
h2_fea = []
h0_lab = []
h0_local_lab = []
h1_lab = []
h1_local_lab = []
h2_lab = []

with torch.no_grad():
    for _, (images, h0_labels, h1_labels, h2_labels, h0_local_labels, h1_local_labels) in enumerate(test_loader):
        images = Variable(images.cuda())
        content, style = encode(images)
        batch_size = images.size(0)

        content_fea.append(content)
        h0_fea.append(style[0])
        h1_fea.append(style[1])
        h0_lab.append(h0_labels)
        h0_local_lab.append(h0_local_labels)
        h1_lab.append(h1_labels)
        h1_local_lab.append(h1_local_labels)
        h2_lab.append(h2_labels)
        if trainer.root_num>1:
            h2_fea.append(style[2])
        count = count + batch_size
        if count >= sample_num or count >= max_num:
            break

    h0_fea = torch.cat(h0_fea).cpu().numpy()
    h1_fea = torch.cat(h1_fea).cpu().numpy()
    if trainer.root_num>1:
        h2_fea = torch.cat(h2_fea).cpu().numpy()
    content_fea = torch.cat(content_fea).cpu().numpy()
    content_fea = np.reshape(content_fea, [content_fea.shape[0], -1])
    pca = PCA(n_components=50)
    content_fea = pca.fit_transform(content_fea)
    h0_lab = torch.cat(h0_lab).numpy()
    h1_lab = torch.cat(h1_lab).numpy()
    h2_lab = torch.cat(h2_lab).numpy()
    h0_local_lab = torch.cat(h0_local_lab).numpy()
    h1_local_lab = torch.cat(h1_local_lab).numpy()

    file_path = os.path.join(output_directory, opts.save_name)

    np.savez(file_path+'.npz', h0_fea=h0_fea, h1_fea=h1_fea, h2_fea=h2_fea, content=content_fea, h0_lab=h0_lab, h1_lab=
             h1_lab, h2_lab=h2_lab, h0_local_lab=h0_local_lab, h1_local_lab=h1_local_lab)
    sio.savemat(file_path+'.mat', {'h0_fea':h0_fea, 'h1_fea':h1_fea, 'h2_fea':h2_fea, 'content':content_fea, 'h0_lab':
                                   h0_lab, 'h1_lab':h1_lab, 'h2_lab':h2_lab, 'h0_local_lab':h0_local_lab,
                                   'h1_local_lab':h1_local_lab})

