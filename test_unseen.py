from __future__ import print_function
from utils import get_config, get_data_loader_list, _count_hierarchy_pred
from trainer import HDN_Trainer
import torch.backends.cudnn as cudnn
import argparse
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import torch
import os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/cadcars.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--unseen_file', type=str, help="unseen image list file")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--output_only', action='store_true', help="whether output raw input images")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
np.random.seed(opts.seed)
cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
# Setup model and data loader
trainer = HDN_Trainer(config)

batch_size = config['batch_size']
num_workers = config['num_workers']
root = config['data_folder']
filter_label = config['filter_labels']
if filter_label == []:
    filter_label = None
tree = config['gen']['tree']
if 'new_size' in config:
    new_size_a = config['new_size']
    new_size_b = config['new_size']
else:
    new_size_a = config['new_size_a']
    new_size_b = config['new_size_b']
height = config['crop_image_height']
width = config['crop_image_width']
crop = False if height == new_size_a and width == new_size_b else True

test_loader = get_data_loader_list(root, config['test_list_file'], filter_label, tree, batch_size, False, new_size_a,
                                   height, width, num_workers, crop, False)
unseen_root = '/home/qiaoshishi/datasets/CompCars/re-organize_compcars_seg/'
unseen_tree = [4, 3, 3, 3, 3, 1, 4]
# SUV, SEDAN, HATCHBACK, MINIBUS, PICKUP, CONVERTIBLE
# 185, 228, 101,    416, 324, 315,   546, 632, 592,  725, 738, 754,    808, 814, 817,    917, 924, 915
unseen_filter_label = [725, 738, 754, 185, 228, 101, 917, 924, 915, 416, 324, 315]
unseen_new_size_a = 128
unseen_height = 128
unseen_width = 128
unseen_crop = False if unseen_height == unseen_new_size_a and unseen_width==unseen_new_size_a else True
test_unseen_loader = get_data_loader_list(unseen_root, opts.unseen_file, unseen_filter_label, unseen_tree, batch_size, False,
                                        unseen_new_size_a, unseen_height, unseen_width, num_workers, unseen_crop, False)

model_name = config['model_name']
output_directory1 = os.path.join("./results", model_name, opts.unseen_file, 'single_layer/')
output_directory2 = os.path.join("./results", model_name, opts.unseen_file, 'two_layers/')
output_directory3 = os.path.join("./results", model_name, opts.unseen_file, 'three_layers/')

if not os.path.exists(output_directory1):
    print("Creating directory: {}".format(output_directory1))
    os.makedirs(output_directory1)
if not os.path.exists(output_directory2):
    print("Creating directory: {}".format(output_directory2))
    os.makedirs(output_directory2)
if not os.path.exists(output_directory3):
    print("Creating directory: {}".format(output_directory3))
    os.makedirs(output_directory3)

state_dict_gen = torch.load(opts.checkpoint)
trainer.gen.load_state_dict(state_dict_gen['gen'])

trainer.cuda()
trainer.eval()
encode = trainer.gen.encode

# Start testing
max_test_num = 500
test_count, unseen_count = 0, 0
test_sample_num, unseen_sample_num = test_loader.dataset.__len__(), test_unseen_loader.dataset.__len__()
correct_test_h0, correct_test_h1, correct_test_h2, correct_unseen_h0, correct_unseen_h1, correct_unseen_h2 = 0., 0., 0., \
                                                                                                             0., 0., 0.
entropy_test_h0, entropy_unseen_h0 = 0., 0.
correct_unseen_h1_labels, unseen_leaf_local_pred = list([]), list([])

def rand_h0_local_labels(parent_labs):
    unseen_local_labels = torch.zeros_like(parent_labs)
    offset = 5
    for it, p_lab in enumerate(parent_labs):
        unseen_local_labels[it] = offset
    return unseen_local_labels

with torch.no_grad():
    for _, (test_unseen_data, test_data) in enumerate(zip(test_unseen_loader, test_loader)):

        # hold1 and hold2 are inaccurate generated labels
        unseen_images, hold1, unseen_h1_labels, unseen_h2_labels, hold2, unseen_h1_local_labels = \
            test_unseen_data[0].cuda(), test_unseen_data[1].cuda(), test_unseen_data[2].cuda(), \
            test_unseen_data[3].cuda(), test_unseen_data[4].cuda(), test_unseen_data[5].cuda()
        test_images, test_h0_labels, test_h1_labels, test_h2_labels, test_h0_local_labels, test_h1_local_labels = \
            test_data[0].cuda(), test_data[1].cuda(), test_data[2].cuda(), test_data[3].cuda(), test_data[4].cuda(), \
            test_data[5].cuda(),
        unseen_h0_local_labels = rand_h0_local_labels(unseen_h1_labels).cuda()


        test_images = Variable(test_images.cuda())
        test_content, test_style_real = encode(test_images)
        test_style_real = torch.cat(test_style_real, 1)
        unseen_images = Variable(unseen_images.cuda())
        unseen_content, unseen_style = encode(unseen_images)
        unseen_style = torch.cat(unseen_style, 1)

        # hierarchical accuracy of tested images
        out_test_h0 = trainer.gen.cls_h0(test_style_real)
        out_unseen_h0 = trainer.gen.cls_h0(unseen_style)

        out_test_h1 = trainer.gen.cls_h1(test_style_real)
        out_unseen_h1 = trainer.gen.cls_h1(unseen_style)
        if trainer.dis.h2_classes>1:
            out_test_h2 = trainer.gen.cls_h2(test_style_real)
            out_unseen_h2 = trainer.gen.cls_h2(unseen_style)
            pred_test_h2 = torch.argmax(out_test_h2,1)
            pred_unseen_h2 = torch.argmax(out_unseen_h2,1)
            correct_test_h2 += (pred_test_h2 == test_h2_labels).sum().float()
            correct_unseen_h2 += (pred_unseen_h2 == unseen_h2_labels).sum().float()
            tmp_count, pred_test_h1 = _count_hierarchy_pred(pred_test_h2, test_h2_labels, out_test_h1,
                                                           test_h1_local_labels, tree[tree[0]+1:])
            correct_test_h1 += tmp_count
            tmp_count, pred_unseen_h1 = _count_hierarchy_pred(pred_unseen_h2, unseen_h2_labels, out_unseen_h1,
                                                             unseen_h1_local_labels, tree[tree[0]+1:])
            correct_unseen_h1 += tmp_count
            tmp_count, _, tmp_entropy, _, _ = _count_hierarchy_pred(pred_test_h1, test_h1_labels, out_test_h0,
                                                test_h0_local_labels, tree, True)
            correct_test_h0 += tmp_count
            entropy_test_h0 += tmp_entropy
            tmp_count, _, tmp_entropy, tmp_correct_h1_lab, tmp_leaf_local_pred = _count_hierarchy_pred(
                                                                        pred_unseen_h1, unseen_h1_labels,out_unseen_h0,
                                                                        unseen_h0_local_labels, tree, True)
            correct_unseen_h0 += tmp_count
            entropy_unseen_h0 += tmp_entropy
            correct_unseen_h1_labels += tmp_correct_h1_lab
            unseen_leaf_local_pred += tmp_leaf_local_pred
        else:
            pred_test_h1 = torch.argmax(out_test_h1,1)
            pred_unseen_h1 = torch.argmax(out_unseen_h1,1)
            correct_test_h1 += (pred_test_h1 == test_h1_labels).sum().float()
            correct_unseen_h1 += (pred_unseen_h1 == unseen_h1_labels).sum().float()
            tmp_count, _, tmp_entropy, _, _ = _count_hierarchy_pred(pred_test_h1, test_h1_labels, out_test_h0,
                                                           test_h0_local_labels, tree, True)
            correct_test_h0 += tmp_count
            entropy_test_h0 += tmp_entropy
            tmp_count, _, tmp_entropy, tmp_correct_h1_lab, tmp_leaf_local_pred = _count_hierarchy_pred(pred_unseen_h1,
                                                             unseen_h1_labels, out_unseen_h0,
                                                             unseen_h0_local_labels, tree, True)
            correct_unseen_h0 += tmp_count
            entropy_unseen_h0 += tmp_entropy
            correct_unseen_h1_labels += tmp_correct_h1_lab
            unseen_leaf_local_pred += tmp_leaf_local_pred

        # test the semantic edit for different hierarchy levels
        if test_count >= max_test_num:
            test_count = test_count + test_images.size(0)
            unseen_count = unseen_count + unseen_images.size(0)
            if test_count >= test_sample_num or unseen_count >= unseen_sample_num:
                break
            continue

        images = torch.cat((test_images, unseen_images))
        content = torch.cat((test_content, unseen_content))
        style_real = torch.split(torch.cat((test_style_real, unseen_style)), 8, 1)
        h0_labels = torch.cat((test_h0_labels, hold1))
        h1_labels = torch.cat((test_h1_labels, unseen_h1_labels))
        h2_labels = torch.cat((test_h2_labels, unseen_h2_labels))
        trainer.test_trans(images, content, style_real, output_directory1, test_count, config['w_scale'], config['h_scale'],
                           h0=True, h1=False, h2=False, h0_label=h0_labels, h1_label=h1_labels,  h2_label=h2_labels)

        trainer.test_trans(images, content, style_real, output_directory1, test_count, config['w_scale'], config['h_scale'],
                           h0=False, h1=True, h2=False,
                           h0_label=h0_labels, h1_label=h1_labels,  h2_label=h2_labels)

        if trainer.dis.h2_classes>1:
            trainer.test_trans(images, content, style_real, output_directory1, test_count, config['w_scale'],
                               config['h_scale'], h0=False, h1=False, h2=True,
                               h0_label=h0_labels, h1_label=h1_labels,  h2_label=h2_labels)
            trainer.test_trans(images, content, style_real, output_directory2, test_count, config['w_scale'], config['h_scale'],
                               h0=True, h1=True, h2=False,
                               h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels)
            trainer.test_trans(images, content, style_real, output_directory2, test_count, config['w_scale'], config['h_scale'],
                               h0=True, h1=False, h2=True,
                               h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels)
            trainer.test_trans(images, content, style_real, output_directory2, test_count, config['w_scale'], config['h_scale'],
                               h0=False, h1=True, h2=True,
                               h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels)
            trainer.test_trans(images, content, style_real, output_directory3, test_count, config['w_scale'], config['h_scale'],
                               h0=True, h1=True, h2=True,
                               h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels)
        else:
            trainer.test_trans(images, content, style_real, output_directory3, test_count, config['w_scale'], config['h_scale'],
                               h0=True, h1=True, h2=False,
                               h0_label=h0_labels, h1_label=h1_labels, h2_label=h2_labels)
        test_count = test_count + test_images.size(0)
        unseen_count = unseen_count + unseen_images.size(0)
        if test_count >= test_sample_num or unseen_count >= unseen_sample_num:
            break
    acc_test_h0 = correct_test_h0/test_count*100
    acc_test_h1 = correct_test_h1/test_count*100
    acc_test_h2 = correct_test_h2/test_count*100
    acc_unseen_h0 = correct_unseen_h0/unseen_count*100
    acc_unseen_h1 = correct_unseen_h1/unseen_count*100
    acc_unseen_h2 = correct_unseen_h2/unseen_count*100
    entropy_test_h0 /= correct_test_h1
    entropy_unseen_h0 /= correct_unseen_h1

    print('acc_test_h0:{}\n acc_test_h1:{}\n acc_test_h2:{}\n acc_unseen_h0:{}\n acc_unseen_h1:{}\n acc_unseen_h2:{}\n '
          'entropy_test_h0:{}\n entropy_unseen_h0:{}'.format(acc_test_h0,
          acc_test_h1, acc_test_h2, acc_unseen_h0, acc_unseen_h1, acc_unseen_h2, entropy_test_h0, entropy_unseen_h0))

    np.savez(os.path.join("./results", model_name, opts.unseen_file, 'leaf_pred.npz'),
             h0_local_pred=unseen_leaf_local_pred, correct_h1_labels=correct_unseen_h1_labels)
    sio.savemat(os.path.join("./results", model_name, opts.unseen_file, 'leaf_pred.mat'),
                {'h0_local_pred': unseen_leaf_local_pred, 'correct_h1_labels': correct_unseen_h1_labels})
    sys.exit('Finish test')

