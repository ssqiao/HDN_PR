from __future__ import print_function
from utils import get_config, count_hierarchy_pred
from trainer import HDN_Trainer
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import numpy as np
import utils as utl
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os, sys

#
# Main features: note the input value range for different features ([-1,1], [0,255] or any others)
# 1. style transfer between two input images
# 2. interpolate between styles of two input images
# 3. test LPIPS, Acc
# 4. test IS and FID
# LPIPS, FID and IS are tested in independent project (perceptive similarity) or codes (FID, IS)
#

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/celeba.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint_gen', type=str, help="checkpoint of autoencoders")
parser.add_argument('--checkpoint_dis', type=str, help="checkpoint of discriminators")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--output_only', action='store_true', help="whether output raw input images")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
tree = config['gen']['tree']
num_scales = config['dis']['num_scales']
# Setup model and data loader
trainer = HDN_Trainer(config)

_, test_loader = utl.get_all_data_loaders(config)

model_name = config['model_name']
output_directory_input = os.path.join("./results", model_name, 'input/')
output_directory_gen_all = os.path.join("./results", model_name, 'translated/')
output_directory1 = os.path.join("./results", model_name, 'single_layer/')
output_directory2 = os.path.join("./results", model_name, 'two_layers/')
output_directory3 = os.path.join("./results", model_name, 'three_layers/')

if not os.path.exists(output_directory_input):
    print("Creating directory: {}".format(output_directory_input))
    os.makedirs(output_directory_input)
if not os.path.exists(output_directory_gen_all):
    print("Creating directory: {}".format(output_directory_gen_all))
    os.makedirs(output_directory_gen_all)
if not os.path.exists(output_directory1):
    print("Creating directory: {}".format(output_directory1))
    os.makedirs(output_directory1)
if not os.path.exists(output_directory2):
    print("Creating directory: {}".format(output_directory2))
    os.makedirs(output_directory2)
if not os.path.exists(output_directory3):
    print("Creating directory: {}".format(output_directory3))
    os.makedirs(output_directory3)

state_dict_gen = torch.load(opts.checkpoint_gen)
trainer.gen.load_state_dict(state_dict_gen['gen'])
state_dict_dis = torch.load(opts.checkpoint_dis)
trainer.dis.load_state_dict(state_dict_dis['dis'])


trainer.cuda()
trainer.eval()
encode = trainer.gen.encode
discriminator = trainer.dis

# Start testing
max_test_num = 500
count = 0
sample_num = test_loader.dataset.__len__()
correct_test_h0, correct_test_h1, correct_test_h2 = 0., 0., 0.
correct_trans_h0, correct_trans_h1, correct_trans_h2 = 0., 0., 0.

with torch.no_grad():
    for _, (images, h0_labels, h1_labels, h2_labels, h0_local_labels, h1_local_labels) in enumerate(test_loader):
        images = Variable(images.cuda())
        h0_labels  = h0_labels.cuda()
        h1_labels = h1_labels.cuda()
        h2_labels = h2_labels.cuda()
        h0_local_labels = h0_local_labels.cuda()
        h1_local_labels = h1_local_labels.cuda()
        content, style_real = encode(images)
        batch_size = images.size(0)

        if not opts.output_only:
            for i in range(images.size(0)):
                vutils.save_image(images[i].data, os.path.join(output_directory_input, '{:03d}_{:02d}_{:02d}_{:02d}.jpg'.
                                                               format(count+i, h0_labels[i], h1_labels[i], h2_labels[i])),
                                  padding=0, normalize=True)

        # hierarchical acc of tested real images
        # tar labels only used for ask for memory
        perm_idx = range(batch_size - 1, -1, -1)
        _, test_outs_h0, test_outs_h1, test_outs_h2 = discriminator.forward(images)
        tmp_test_h0, tmp_test_h1,  tmp_test_h2 = count_hierarchy_pred(test_outs_h0, test_outs_h1, test_outs_h2,
                                                                      h0_local_labels, h1_labels, h1_local_labels,
                                                                      h2_labels, h2_labels, tree, cascade=False)
        correct_test_h0 += tmp_test_h0
        correct_test_h1 += tmp_test_h1
        correct_test_h2 += tmp_test_h2

        x_trans = trainer.test_trans(images, content, style_real, output_directory1, count, config['w_scale'],
                                     config['h_scale'], h0=True, h1=False, h2=False, h0_label=h0_labels,
                                     h1_label=h1_labels,  h2_label=h2_labels, ret_trans=True)

        _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
        tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                  h0_local_labels[perm_idx], h1_labels[perm_idx],
                                                                  h1_local_labels, h2_labels, h2_labels, tree,
                                                                  cascade=False)
        for i in range(x_trans.size(0)):
            vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                            '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                            (count+i, count+perm_idx[i], 1, 0, 0)), padding=0, normalize=True)
        correct_trans_h0 += tmp_tar_h0
        correct_trans_h1 += tmp_tar_h1
        correct_trans_h2 += tmp_tar_h2

        x_trans = trainer.test_trans(images, content, style_real, output_directory1, count, config['w_scale'],
                                     config['h_scale'], h0=False, h1=True, h2=False,
                                     h0_label=h0_labels, h1_label=h1_labels,  h2_label=h2_labels, ret_trans=True)
        _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
        tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                  h0_local_labels, h1_labels, h1_local_labels[perm_idx],
                                                                  h2_labels[perm_idx], h2_labels, tree,
                                                                  cascade=False)
        for i in range(x_trans.size(0)):
            vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                            '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                            (count+i, count+perm_idx[i], 0, 1, 0)), padding=0, normalize=True)
        correct_trans_h0 += tmp_tar_h0
        correct_trans_h1 += tmp_tar_h1
        correct_trans_h2 += tmp_tar_h2

        if trainer.dis.h2_classes>1:
            x_trans = trainer.test_trans(images, content, style_real, output_directory1, count, config['w_scale'],
                                         config['h_scale'], h0=False, h1=False, h2=True,
                                         h0_label=h0_labels, h1_label=h1_labels,  h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels, h1_labels,
                                                                      h1_local_labels,
                                                                      h2_labels, h2_labels[perm_idx], tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 0, 0, 1)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2
            x_trans = trainer.test_trans(images, content, style_real, output_directory2, count, config['w_scale'],
                                         config['h_scale'], h0=True, h1=True, h2=False,
                                         h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels[perm_idx], h1_labels[perm_idx],
                                                                      h1_local_labels[perm_idx],
                                                                      h2_labels[perm_idx], h2_labels, tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 1, 1, 0)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2
            x_trans = trainer.test_trans(images, content, style_real, output_directory2, count, config['w_scale'],
                                         config['h_scale'], h0=True, h1=False, h2=True,
                                         h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels[perm_idx], h1_labels[perm_idx],
                                                                      h1_local_labels,
                                                                      h2_labels, h2_labels[perm_idx], tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 1, 0, 1)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2
            x_trans = trainer.test_trans(images, content, style_real, output_directory2, count, config['w_scale'],
                                         config['h_scale'], h0=False, h1=True, h2=True,
                                         h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels, h1_labels,
                                                                      h1_local_labels[perm_idx],
                                                                      h2_labels[perm_idx], h2_labels[perm_idx], tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 0, 1, 1)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2
            x_trans = trainer.test_trans(images, content, style_real, output_directory3, count, config['w_scale'],
                                         config['h_scale'], h0=True, h1=True, h2=True,
                                         h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels[perm_idx], h1_labels[perm_idx],
                                                                      h1_local_labels[perm_idx],
                                                                      h2_labels[perm_idx], h2_labels[perm_idx], tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 1, 1, 1)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2
        else:
            x_trans = trainer.test_trans(images, content, style_real, output_directory3, count, config['w_scale'],
                                         config['h_scale'], h0=True, h1=True, h2=False,
                                         h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels, ret_trans=True)
            _, tar_outs_h0, tar_outs_h1, tar_outs_h2 = discriminator.forward(x_trans)
            tmp_tar_h0, tmp_tar_h1, tmp_tar_h2 = count_hierarchy_pred(tar_outs_h0, tar_outs_h1, tar_outs_h2,
                                                                      h0_local_labels[perm_idx], h1_labels[perm_idx],
                                                                      h1_local_labels[perm_idx],
                                                                      h2_labels[perm_idx], h2_labels, tree,
                                                                      cascade=False)
            for i in range(x_trans.size(0)):
                vutils.save_image(x_trans[i].data, os.path.join(output_directory_gen_all,
                                                                '{:03d}_{:03d}_h0_{:01d}_h1_{:01d}_h2_{:01d}.jpg'.format
                                                                (count + i, count + perm_idx[i], 1, 1, 0)), padding=0,
                                  normalize=True)
            correct_trans_h0 += tmp_tar_h0
            correct_trans_h1 += tmp_tar_h1
            correct_trans_h2 += tmp_tar_h2

        count = count + batch_size
        if count >= max_test_num or count >= sample_num:
            break

    acc_test_h0 = correct_test_h0/count/num_scales*100
    acc_test_h1 = correct_test_h1/count/num_scales*100
    acc_test_h2 = correct_test_h2/count/num_scales*100
    acc_trans_h0 = correct_trans_h0/count/num_scales/7*100 if trainer.dis.h2_classes > 1 else \
                   correct_trans_h0/count/num_scales/3*100
    acc_trans_h1 = correct_trans_h1/count/num_scales/7*100 if trainer.dis.h2_classes > 1 else \
                   correct_trans_h1/count/num_scales/3*100
    acc_trans_h2 = correct_trans_h2/count/num_scales/7*100 if trainer.dis.h2_classes > 1 else \
                   correct_trans_h2/count/num_scales/3*100
    print('acc_test_h0:{}\n acc_test_h1:{}\n acc_test_h2:{}\n acc_trans_h0:{}\n acc_trans_h1:{}\n acc_trans_h2:{}'.
          format(acc_test_h0, acc_test_h1, acc_test_h2, acc_trans_h0, acc_trans_h1, acc_trans_h2))
    sys.exit('Finish test')

