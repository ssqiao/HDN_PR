from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageLabelFilelist
from scipy import misc
import torch
import torch.nn.functional as F
import os
import math
import yaml
import numpy as np
import torch.nn.init as init
import time
# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_config                : load yaml file
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_loss                : write records into tensorboard
# get_model_list            : Get model list for resume
# get_scheduler             : set the learning rate scheduler for optimization
# weights_init              : get init func for network weights


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    root = conf['data_folder']
    filter_label = conf['filter_labels']
    if filter_label == []:
        filter_label = None
    tree = conf['gen']['tree']
    train_file = conf['train_list_file']
    test_file = conf['test_list_file']
    if 'new_size' in conf:
        new_size_a = conf['new_size']
        new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
    crop = False if height == new_size_a and width == new_size_b else True
    horizon = True if 'multipie' not in conf['model_name'] and 'cadcars' not in conf['model_name'] else False

    train_data_loader = get_data_loader_list(root, train_file, filter_label, tree, batch_size, True,
                                             new_size_a, height, width, num_workers, crop, horizon)
    test_data_loader = get_data_loader_list(root, test_file, filter_label, tree, batch_size, False,
                                            new_size_a, height, width, num_workers, crop, horizon)
    return train_data_loader, test_data_loader


def get_data_loader_list(root, file_list, filter_label, tree, batch_size, train, new_size=None,
                         height=128, width=128, num_workers=4, crop=True, horizon=True, pre_shuffle=True):
    transform_list = [transforms.ToTensor(),  # [0,255] to [0,1]
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]  # [0,1] to [-1,1]
    transform_list = [transforms.Resize(
        (new_size, new_size))] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.CenterCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train and horizon else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageLabelFilelist(root, file_list, filter_label, tree, transform=transform, shuffle=pre_shuffle)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

# qss, modify the dimension acc. to the size of image_outputs in each line
def _write_images(image_outputs, file_name, w_scale, h_scale):
    # expand gray-scale images to 3 channels, (b,c,w,h)
    image_outputs = image_outputs.expand(-1, 3, -1, -1) if image_outputs.size(1) == 1 else image_outputs
    m = 10
    canvas = 255*np.ones((image_outputs.size(2)//h_scale,
                          image_outputs.size(0)*image_outputs.size(3)//w_scale+(m*image_outputs.size(0))+m, 3),
                         dtype=np.uint8)

    # resize the w/h ratio
    start = m
    for img in image_outputs:
        img = img.cpu().numpy()
        img = (img + 1.)
        img *= 127.5
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = img.transpose((1, 2, 0))
        shape = np.shape(img)
        img = misc.imresize(img, (shape[0]//h_scale, shape[1]//w_scale))
        end = start + shape[1]
        canvas[:, start:end, :] = img
        start = end + m

    misc.imsave(file_name, canvas)


# qss, display_image_num: style sample num for each source image in each hierarchy node, i.e. saved image # in each row
def write_2images(img_raw, img_tar, img_trans, heights, label_a, label_b, image_directory, postfix,
                  w_scale, h_scale):
    display_size = img_raw.size(0)
    height_num = len(heights)
    for i in range(display_size):
        raw = img_raw[i].unsqueeze(0)
        tar = img_tar[i].unsqueeze(0)
        trans = img_trans[i:(i+1)]
        imgs = torch.cat((raw,trans,tar))
        h0 = heights[0][i]
        h1 = heights[1][i]
        h2 = 0
        local_lab_a0 = label_a[0][i]
        lab_a0 = label_a[1][i]
        local_lab_b0 = label_b[0][i]
        lab_b0 = label_b[1][i]
        local_lab_a1 = label_a[2][i]
        lab_a1 = label_a[3][i]
        local_lab_b1 = label_b[2][i]
        lab_b1 = label_b[3][i]

        if height_num>2:
            h2 = heights[2][i]
            lab_a2 = label_a[3][i]
            lab_b2 = label_b[4][i]

        if h0+h1+h2 == 0:
            str_h = '_h-1'
            _write_images(imgs,
                          '%s/gen_%s.jpg' % (image_directory, postfix + str(i) + str_h), w_scale, h_scale)

        elif h0+h1+h2 == 1:
            if h0:
                str_h = '_h0'
                _write_images(imgs,
                              '%s/gen_%s_p%d_c%dTop%d_c%d.jpg' % (image_directory, postfix + str(i) + str_h, lab_a0,
                                                                  local_lab_a0, lab_b0, local_lab_b0), w_scale, h_scale)
            elif h1:
                str_h = '_h1'
                _write_images(imgs,
                              '%s/gen_%s_p%d_c%dTop%d_c%d.jpg' % (image_directory, postfix + str(i) + str_h, lab_a1,
                                                                  local_lab_a1, lab_b1, local_lab_b1), w_scale, h_scale)
            else:
                str_h = '_h2'
                _write_images(imgs,
                              '%s/gen_%s_%dTo%d.jpg' % (image_directory, postfix + str(i) + str_h, lab_a2,
                                                                   lab_b2), w_scale, h_scale)


        elif h0+h1+h2 == 2:
            if h0 and h1:
                str_h = '_h0h1'
                _write_images(imgs,
                              '%s/gen_%s_p%d_c%dTop%d_c%d_p%d_c%dTop%d_c%d.jpg' %
                              (image_directory, postfix + str(i) + str_h, lab_a0, local_lab_a0, lab_b0, local_lab_b0,
                                lab_a1, local_lab_a1, lab_b1, local_lab_b1), w_scale, h_scale)
            elif h0 and h2:
                str_h = '_h0h2'
                _write_images(imgs,
                              '%s/gen_%s_p%d_c%dTop%d_c%d_%dTo%d.jpg' %
                              (image_directory, postfix + str(i) + str_h, lab_a0, local_lab_a0, lab_b0, local_lab_b0,
                               lab_a2, lab_b2), w_scale, h_scale)
            else:
                str_h = '_h1h2'
                _write_images(imgs,
                              '%s/gen_%s_p%d_c%dTop%d_c%d_%dTo%d.jpg' %
                              (image_directory, postfix + str(i) + str_h, lab_a1, local_lab_a1, lab_b1, local_lab_b1,
                               lab_a2, lab_b2), w_scale, h_scale)


        else:
            str_h = '_h0h1h2'
            _write_images(imgs,
                          '%s/gen_%s_p%d_c%dTop%d_c%d_p%d_c%dTop%d_c%d_%dTo%d.jpg' %
                          (image_directory, postfix + str(i) + str_h, lab_a0, local_lab_a0, lab_b0, local_lab_b0,
                           lab_a1, local_lab_a1, lab_b1, local_lab_b1, lab_a2, lab_b2), w_scale, h_scale)

# compute num of current level's accurate predictions (and classification entropy if entropy)
# if cascade, then predictions of two adjacent levels are related, i.e.,
# if parent pred failed, its child pred is ignored (this is for real images, False for translated images)
def _count_hierarchy_pred(parent_pred, parent_labels, local_out, local_labels, tree, entropy=False, cascade=True):
    count = 0
    info_entropy = 0.
    current_pred = list([])
    leaf_local_pred = list([])
    correct_parent_lab = list([])

    eps = 1e-12
    for it, (p_pred, p_lab, l_out, l_lab) in enumerate(zip(parent_pred, parent_labels, local_out, local_labels)):
        l_split = torch.split(l_out, tree[1:tree[0]+1]) # dim
        if cascade:
            l_pred = torch.argmax(l_split[p_pred]) # dim
            if p_pred == p_lab and l_pred == l_lab:
                count += 1
        else:
            l_pred = torch.argmax(l_split[p_lab]) # dim
            if l_pred == l_lab:
                count += 1
        current_pred.append(sum(tree[1:p_pred + 1]) + l_pred)

        if entropy and p_pred==p_lab:
            leaf_local_pred.append(l_pred)
            correct_parent_lab.append(p_lab)
            child = F.softmax(l_split[p_lab])
            log_child = torch.log(child + eps)
            element_entropy = child * log_child
            info_entropy += torch.sum(element_entropy) # dim
    if not entropy:
        return count, current_pred
    else:
        return count, current_pred, info_entropy, correct_parent_lab, leaf_local_pred

def count_hierarchy_pred(outs_h0, outs_h1, outs_h2, h0_local_labels, h1_labels, h1_local_labels, h2_labels,
                         h2_local_labels, tree, cascade=False):
    h2_classes = tree[tree[0]+1]
    correct_h0, correct_h1, correct_h2 = 0, 0, 0
    if h2_classes>1:
        for it, (out_h0, out_h1, out_h2) in enumerate(zip(outs_h0, outs_h1, outs_h2)):
            pred_h2 = torch.argmax(out_h2, 1)
            correct_h2 += (pred_h2 == h2_local_labels).sum().float()
            tmp_count, pred_h1 = _count_hierarchy_pred(pred_h2, h2_labels, out_h1, h1_local_labels, tree[tree[0]+1:],
                                                       cascade=cascade)
            correct_h1 += tmp_count
            tmp_count, _ = _count_hierarchy_pred(pred_h1, h1_labels, out_h0, h0_local_labels, tree, cascade=cascade)
            correct_h0 += tmp_count
    else:
        for it, (out_h0, out_h1) in enumerate(zip(outs_h0, outs_h1)):
            pred_h1 = torch.argmax(out_h1, 1)
            correct_h1 += (pred_h1 == h1_local_labels).sum().float()
            tmp_count, _ = _count_hierarchy_pred(pred_h1, h1_labels, out_h0, h0_local_labels, tree, cascade=cascade)
            correct_h0 += tmp_count
    return correct_h0, correct_h1, correct_h2

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) if not callable(getattr(trainer, attr)) and not attr.startswith("__") and
               ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
