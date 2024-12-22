from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, get_scheduler, _write_images
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from data import get_super_category


class HDN_Trainer(nn.Module):
    def __init__(self, hyperparameters):  # qss, init with a hyperparameters dict, one AE & Msdis
        super(HDN_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis = MsImageDis(hyperparameters['input_dim'], hyperparameters['dis'])  # discriminator for domain a

        self.root_num = self.gen.tree[self.gen.tree[0]+1]

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Applies ``fn`` recursively to every submodule Network weight initialization, gen use kaiming, dis use gaussian
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def total_variation_loss(self, x):
        assert x.ndimension() == 4

        a = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        b = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

        return a+b

    # style transfer
    def forward(self, x1, x2):
        self.eval()
        c1, s1 = self.gen.encode(x1.unsqueeze(0))
        c2, s2 = self.gen.encode(x2.unsqueeze(0))
        s1 = torch.cat(s1, 1)
        s2 = torch.cat(s2, 1)
        x_ab = self.gen.decode(c1, s2)
        x_ba = self.gen.decode(c2, s1)
        self.train()
        return x_ab, x_ba

    def permute_recompostion(self, styles, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label=None):
        output_styles = list([])
        # change_heights = list([])
        batch_size = h1_label.size(0)
        rand_idx = torch.randperm(batch_size)
        src_h0 = styles[0]
        src_h1 = styles[1]
        tar_local_lab_h0 = h0_local_label.clone().detach()
        tar_lab_h1 = h1_label.clone().detach()
        tar_local_lab_h1 = h1_local_label.clone().detach()
        tar_lab_h2 = h2_label.clone().detach()
        h0 = torch.randint(low=0, high=2, size=(batch_size,), dtype=torch.int32)
        h1 = torch.randint(low=0, high=2, size=(batch_size,), dtype=torch.int32)
        # change_heights = [h0, h1]

        if self.root_num>1:
            src_h2 = styles[2]
            tar_local_lab_h2 = h2_local_label.clone().detach()
            h2 = torch.randint(low=0, high=2, size=(batch_size,), dtype=torch.int32)
            # h2 = torch.zeros(batch_size)
            # change_heights = [h0, h1, h2]

        for i in range(batch_size):
            if h0[i] and h1[i]:
                tmp_style = torch.cat((src_h0[rand_idx[i]], src_h1[rand_idx[i]]))
                tar_local_lab_h0[i] = h0_local_label[rand_idx[i]]
                tar_lab_h1[i] = h1_label[rand_idx[i]]
                tar_local_lab_h1[i] = h1_local_label[rand_idx[i]]
                tar_lab_h2[i] = h2_label[rand_idx[i]]
            elif h0[i] and not h1[i]:
                tmp_style = torch.cat((src_h0[rand_idx[i]], src_h1[i]))
                tar_local_lab_h0[i] = h0_local_label[rand_idx[i]]
                tar_lab_h1[i] = h1_label[rand_idx[i]]
            elif not h0[i] and h1[i]:
                tmp_style = torch.cat((src_h0[i], src_h1[rand_idx[i]]))
                tar_local_lab_h1[i] = h1_local_label[rand_idx[i]]
                tar_lab_h2[i] = h2_label[rand_idx[i]]
            elif not h0[i] and not h1[i]:
                tmp_style = torch.cat((src_h0[i], src_h1[i]))
            if self.root_num>1:
                if h2[i]:
                    tmp_style = torch.cat((tmp_style, src_h2[rand_idx[i]]))
                    tar_local_lab_h2[i] = h2_local_label[rand_idx[i]]
                else:
                    tmp_style = torch.cat((tmp_style, src_h2[i]))
            output_styles.append(tmp_style.unsqueeze(0))

        if self.root_num>1:
            return torch.cat(output_styles), rand_idx, (h0,h1,h2), tar_local_lab_h0, tar_lab_h1, tar_local_lab_h1, tar_lab_h2,\
                   tar_local_lab_h2
        else:
            return torch.cat(output_styles), rand_idx, (h0,h1), tar_local_lab_h0, tar_lab_h1, tar_local_lab_h1, tar_lab_h2

    # qss
    def gen_update(self, x, hyperparameters, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label=None):
        self.gen_opt.zero_grad()

        # encode
        c, s_prime = self.gen.encode(x)

        if self.root_num>1:
            s_composite, _, _, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels, tar_h2_local_labels \
                = self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label)
        else:
            s_composite, _, _, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels = \
                self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label)
            tar_h2_local_labels = None


        # decode (within domain), use style from real image
        s_prime = torch.cat(s_prime, 1)
        x_recon = self.gen.decode(c, s_prime)

        # decode (cross domain), use style from a distribution
        x_trans = self.gen.decode(c, s_composite)

        # encode again
        c_recon, s_composite_recon = self.gen.encode(x_trans)
        s_composite_recon = torch.cat(s_composite_recon, 1)

        # reconstruction loss
        self.loss_gen_recon_x = self.recon_criterion(x_recon, x)

        self.loss_gen_recon_c = self.recon_criterion(c, c_recon)

        self.loss_gen_recon_s = self.recon_criterion(s_composite, s_composite_recon)

        self.loss_gen_tv = self.total_variation_loss(x_trans) \
            if 'tv_w' in hyperparameters.keys() and hyperparameters['tv_w'] > 0 else 0.

        self.loss_gen_adv, self.loss_gen_cls_h0, self.loss_gen_cls_h1, self.loss_gen_cls_h2 = \
            self.dis.calc_gen_loss(x_trans, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels,
                                   tar_h2_local_labels)

        # information entropy loss for disentangled features
        self.loss_gen_disent_cls_prim, self.loss_gen_disent_entropy_prim = \
            self.compute_disent_loss(s_prime, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label)
        self.loss_gen_disent_cls_comp, self.loss_gen_disent_entropy_comp = \
            self.compute_disent_loss(s_composite,tar_h0_local_labels, tar_h1_labels,
                                     tar_h1_local_labels, tar_h2_labels, tar_h2_local_labels)

        # total loss#
        self.loss_gen_total = hyperparameters['gan_w'] * \
            (self.loss_gen_adv + 1.0*(self.loss_gen_cls_h0 + self.loss_gen_cls_h1 + self.loss_gen_cls_h2)) + \
            hyperparameters['disent_cls_w'] * (self.loss_gen_disent_cls_prim + self.loss_gen_disent_cls_comp) + \
            hyperparameters['disent_ent_w'] * (self.loss_gen_disent_entropy_prim + self.loss_gen_disent_entropy_comp) + \
            hyperparameters['recon_x_w'] * self.loss_gen_recon_x + \
            hyperparameters['recon_s_w'] * self.loss_gen_recon_s + \
            hyperparameters['recon_c_w'] * self.loss_gen_recon_c + \
            hyperparameters['tv_w'] * self.loss_gen_tv

        self.loss_gen_total.backward()
        self.gen_opt.step()

    # Compute the entropy of predicted category probabilities
    def negative_entropy_prediction(self, logits, tree=None):
        eps = 1e-12
        # entropy of all categories in the level
        if tree is None:
            first_child = F.softmax(logits)
            log_first_child = torch.log(first_child + eps)
            element_entropy = first_child * log_first_child
            first_child_entropy = torch.sum(element_entropy, 1).unsqueeze(1)
            entropy = first_child_entropy
            return entropy

        # entropy of local brothers for each super category
        num_supercategory = tree[0]

        splited_logits = torch.split(logits, tree[1:num_supercategory + 1], dim=-1)

        first_child = F.softmax(splited_logits[0])
        log_first_child = torch.log(first_child + eps)
        element_entropy = first_child * log_first_child
        first_child_entropy = torch.sum(element_entropy, 1).unsqueeze(1)
        entropy = first_child_entropy

        for i in range(1, num_supercategory):
            tmp_child = F.softmax(splited_logits[i])
            log_tmp_child = torch.log(tmp_child + eps)
            tmp_element_entropy = tmp_child * log_tmp_child
            tmp_child_entropy = torch.sum(tmp_element_entropy, 1).unsqueeze(1)
            entropy = torch.cat((entropy, tmp_child_entropy), 1)

        return entropy

    def compute_local_cls_loss(self, logits, local_idx, class_num, local_labels, parent_labels):
        cls_local_loss = 0.
        local_logits = torch.split(logits, local_idx, 1)
        for lab in range(class_num):
            tmp = torch.eq(parent_labels, lab)
            flag = torch.sum(tmp)
            if flag > 0:
                idx = torch.nonzero(tmp)
                select_local_labels = local_labels[idx].squeeze(1)
                select_local_logits = local_logits[lab][idx,:].squeeze(1)
                cls_local_loss += torch.sum(F.cross_entropy(select_local_logits, select_local_labels))

        return cls_local_loss/logits.size(0)

    def compute_disent_loss(self, styles, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label=None):

        s_tmp = torch.split(styles, 8, 1)

        s_h0 = s_tmp[0]
        s_h1 = s_tmp[1]
        s_pad_h0 = torch.zeros_like(s_h0).cuda()
        s_pad_h1 = torch.zeros_like(s_h1).cuda()
        s_h0_diff = torch.cat((s_h0, s_pad_h1), 1)
        s_h1_diff = torch.cat((s_pad_h0, s_h1), 1)
        if self.root_num>1 and h2_label is not None:
            s_h2 = s_tmp[2]
            s_pad_h2 = torch.zeros_like(s_h2)
            s_h2_diff = torch.cat((s_pad_h0, s_pad_h1, s_h2), 1)
            s_h1_diff = torch.cat((s_h1_diff, s_pad_h2),1)
            s_h0_diff = torch.cat((s_h0_diff, s_pad_h2), 1)
            s_h1_h2_diff = torch.cat((s_pad_h0, s_h1, s_h2), 1)

            fea_h0_min_pred_h0 = self.gen.cls_h0(styles)
            fea_h0_min_pred_h1 = self.gen.cls_h1(styles)
            fea_h0_min_pred_h2 = self.gen.cls_h2(styles)
            fea_h1_min_pred_h0 = self.gen.cls_h0(s_h1_h2_diff)
            fea_h1_min_pred_h1 = self.gen.cls_h1(s_h1_h2_diff)
            fea_h1_min_pred_h2 = self.gen.cls_h2(s_h1_h2_diff)
            fea_h2_min_pred_h0 = self.gen.cls_h0(s_h2_diff)
            fea_h2_min_pred_h1 = self.gen.cls_h1(s_h2_diff)
            fea_h2_min_pred_h2 = self.gen.cls_h2(s_h2_diff)
            fea_h1_diff_pred_h0 = self.gen.cls_h0(s_h1_diff)
            fea_h1_diff_pred_h1 = self.gen.cls_h1(s_h1_diff)
            fea_h1_diff_pred_h2 = self.gen.cls_h2(s_h1_diff)
            fea_h0_diff_pred_h0 = self.gen.cls_h0(s_h0_diff)
            fea_h0_diff_pred_h1 = self.gen.cls_h1(s_h0_diff)
            fea_h0_diff_pred_h2 = self.gen.cls_h2(s_h0_diff)

            loss_cls = torch.mean(F.cross_entropy(fea_h2_min_pred_h2, h2_local_label)) \
                       + self.compute_local_cls_loss(fea_h1_diff_pred_h1, self.gen.tree[self.gen.tree[0]+2:],
                                                     self.gen.root_num, h1_local_label, h2_label) \
                       + self.compute_local_cls_loss(fea_h0_diff_pred_h0, self.gen.tree[1:self.gen.tree[0]+1],
                                                     self.gen.intermediate_num, h0_local_label, h1_label) \
                       + torch.mean(F.cross_entropy(fea_h1_min_pred_h2, h2_local_label)) \
                       + self.compute_local_cls_loss(fea_h1_min_pred_h1, self.gen.tree[self.gen.tree[0]+2:],
                                                     self.gen.root_num, h1_local_label, h2_label) \
                       + torch.mean(F.cross_entropy(fea_h0_min_pred_h2, h2_local_label)) \
                       + self.compute_local_cls_loss(fea_h0_min_pred_h1, self.gen.tree[self.gen.tree[0]+2:],
                                                     self.gen.root_num, h1_local_label, h2_label) \
                       + self.compute_local_cls_loss(fea_h0_min_pred_h0, self.gen.tree[1:self.gen.tree[0]+1],
                                                     self.gen.intermediate_num, h0_local_label, h1_label)

            # non-discriminative among each direct child branch of offsprings
            entropy_h1_min_h0 = self.negative_entropy_prediction(fea_h1_min_pred_h0, self.gen.tree)# qss
            split_h1_min_h0 = torch.split(entropy_h1_min_h0, self.gen.tree[self.gen.tree[0]+2:], 1)
            sum_h1_min_h0 = torch.sum(split_h1_min_h0[0], 1).unsqueeze(1) / self.gen.tree[self.gen.tree[0]+2]
            for i in range(1,self.gen.tree[self.gen.tree[0]+1]):
                tmp_h1_min_h0 = torch.sum(split_h1_min_h0[i], 1).unsqueeze(1) / self.gen.tree[self.gen.tree[0]+2+i]
                sum_h1_min_h0 = torch.cat((sum_h1_min_h0, tmp_h1_min_h0), 1)
            h2_one_hot = torch.Tensor(h2_label.size(0), self.dis.h2_classes).cuda().detach()
            h2_one_hot.zero_()
            h2_one_hot.scatter_(1, h2_label.unsqueeze(1), 1)
            entropy_h1_min_h0 = torch.sum(sum_h1_min_h0 * h2_one_hot, 1)
            entropy_h1_min_h0 = torch.mean(entropy_h1_min_h0)
            # entropy_h1_min_h0 = torch.mean(torch.sum(entropy_h1_min_h0, 1)) / self.gen.tree[0]

            entropy_h2_min_h1 = self.negative_entropy_prediction(fea_h2_min_pred_h1,
                                                                 self.gen.tree[self.gen.tree[0] + 1:])
            entropy_h2_min_h1 = torch.mean(torch.sum(entropy_h2_min_h1, 1)) / self.gen.tree[self.gen.tree[0] + 1]
            entropy_h2_min_h0 = self.negative_entropy_prediction(fea_h2_min_pred_h0, self.gen.tree)
            entropy_h2_min_h0 = torch.mean(torch.sum(entropy_h2_min_h0, 1)) / self.gen.tree[0]

            # bottom added fea has no effect on direct ancestors
            entropy_h1_diff_h2 = self.negative_entropy_prediction(fea_h1_diff_pred_h2)
            entropy_h1_diff_h2 = torch.mean(entropy_h1_diff_h2)

            entropy_h1_diff_h0 = self.negative_entropy_prediction(fea_h1_diff_pred_h0, self.gen.tree)#qss
            split_h1_diff_h0 = torch.split(entropy_h1_diff_h0, self.gen.tree[self.gen.tree[0] + 2:], 1)
            sum_h1_diff_h0 = torch.sum(split_h1_diff_h0[0], 1).unsqueeze(1) / self.gen.tree[self.gen.tree[0] + 2]
            for j in range(1, self.gen.tree[self.gen.tree[0] + 1]):
                tmp_h1_diff_h0 = torch.sum(split_h1_diff_h0[j], 1).unsqueeze(1) / self.gen.tree[self.gen.tree[0] + 2 + j]
                sum_h1_diff_h0 = torch.cat((sum_h1_diff_h0, tmp_h1_diff_h0), 1)
            entropy_h1_diff_h0 = torch.sum(sum_h1_diff_h0 * h2_one_hot, 1)
            entropy_h1_diff_h0 = torch.mean(entropy_h1_diff_h0)
            # entropy_h1_diff_h0 = torch.mean(torch.sum(entropy_h1_diff_h0, 1)) / self.gen.tree[0]

            entropy_h0_diff_h1 = self.negative_entropy_prediction(fea_h0_diff_pred_h1, self.gen.tree[self.gen.tree[0]+1:])# qss
            tmp_h2_label = get_super_category(h1_label, self.gen.tree[self.gen.tree[0]+1:])
            tmp_h2_label = torch.from_numpy(tmp_h2_label).cuda()
            tmp_h2_one_hot = torch.Tensor(tmp_h2_label.size(0), self.dis.h2_classes).cuda().detach()
            tmp_h2_one_hot.zero_()
            tmp_h2_one_hot.scatter_(1, tmp_h2_label.unsqueeze(1), 1)
            entropy_h0_diff_h1 = torch.sum(entropy_h0_diff_h1 * tmp_h2_one_hot, 1)
            entropy_h0_diff_h1 = torch.mean(entropy_h0_diff_h1)
            # entropy_h0_diff_h1 = torch.mean(torch.sum(entropy_h0_diff_h1, 1)) / self.gen.tree[self.gen.tree[0]+1]

            entropy_h0_diff_h2 = self.negative_entropy_prediction(fea_h0_diff_pred_h2)
            entropy_h0_diff_h2 = torch.mean(entropy_h0_diff_h2)

            loss_entropy = entropy_h1_min_h0 + entropy_h2_min_h1 + entropy_h2_min_h0 \
                           + entropy_h1_diff_h2 + entropy_h1_diff_h0 + entropy_h0_diff_h1 + entropy_h0_diff_h2



        else:
            fea_h0_min_pred_h0 = self.gen.cls_h0(styles)
            fea_h0_min_pred_h1 = self.gen.cls_h1(styles)
            fea_h1_min_pred_h0 = self.gen.cls_h0(s_h1_diff)
            fea_h1_min_pred_h1 = self.gen.cls_h1(s_h1_diff)
            fea_h0_diff_pred_h0 = self.gen.cls_h0(s_h0_diff)
            fea_h0_diff_pred_h1 = self.gen.cls_h1(s_h0_diff)

            loss_cls = torch.mean(F.cross_entropy(fea_h0_min_pred_h1, h1_local_label)) + \
                       torch.mean(F.cross_entropy(fea_h1_min_pred_h1, h1_local_label)) + \
                       self.compute_local_cls_loss(fea_h0_diff_pred_h0, self.gen.tree[1:self.gen.tree[0]+1],
                                                   self.gen.intermediate_num, h0_local_label, h1_label) + \
                       self.compute_local_cls_loss(fea_h0_min_pred_h0, self.gen.tree[1:self.gen.tree[0]+1],
                                                   self.gen.intermediate_num, h0_local_label, h1_label)
            # top features are non-discriminative among each child branch
            entropy_h1_min_h0 = self.negative_entropy_prediction(fea_h1_min_pred_h0, self.gen.tree)
            entropy_h1_min_h0 = torch.mean(torch.sum(entropy_h1_min_h0,1))/self.gen.tree[0]
            # diff fea of current level has no effect on other levels
            entropy_h0_diff_h1 = self.negative_entropy_prediction(fea_h0_diff_pred_h1)
            entropy_h0_diff_h1 = torch.mean(entropy_h0_diff_h1)

            loss_entropy = entropy_h1_min_h0 + entropy_h0_diff_h1

        return loss_cls, loss_entropy

    # qss, recon & sampled styles to transfer x to several nodes of the tree
    def rand_trans(self, x, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label=None):
        self.eval()

        # encode
        c, s_prime = self.gen.encode(x)

        if self.root_num>1:
            s_composite, rand_idx, levels, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels, \
            tar_h2_local_labels = self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label,
                                                            h2_local_label)
        else:
            s_composite, rand_idx, levels, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels = \
                self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label)
        # s_prime = torch.cat(s_prime, 1)
        # x_recon = self.gen.decode(c, s_prime)
        x_trans = self.gen.decode(c, s_composite)

        self.train()
        if self.root_num>1:
            return rand_idx, x_trans, levels, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels,\
                   tar_h2_local_labels
        else:
            return rand_idx, x_trans, levels, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels

    def test_trans(self, x, c, s, save_path, count, w_s, h_s, h0=True, h1=False, h2=False, h0_label=None,
                   h1_label=None, h2_label=None, ret_trans = False):
        self.eval()
        num = c.size(0)
        # perm_idx = torch.randperm(num)
        perm_idx = range(num-1,-1, -1)

        s_h0 = s[0]
        s_h1 = s[1]
        pad_h0 = torch.zeros_like(s_h0[0])
        pad_h1 = torch.zeros_like(s_h1[0])

        if self.root_num>1:
            s_h2 = s[2]
            pad_h2 = torch.zeros_like(s_h2[0])

        interpolate_num = 5
        x_trans = list([])
        for i in range(num):
            c1 = c[i].unsqueeze(0)
            ca = c1.repeat(interpolate_num, 1, 1, 1)
            if self.root_num>1:
                s1 = torch.cat((s_h0[i], s_h1[i], s_h2[i])).unsqueeze(0)
            else:
                s1 = torch.cat((s_h0[i], s_h1[i])).unsqueeze(0)

            if h0:
                s2 = s_h0[perm_idx[i]]
            else:
                s2 = s_h0[i]
            if h1:
                s2 = torch.cat((s2, s_h1[perm_idx[i]])).unsqueeze(0)
            else:
                s2 = torch.cat((s2, s_h1[i])).unsqueeze(0)
            if self.root_num>1:
                if h2:
                    s2 = torch.cat((s2, s_h2[perm_idx[i]].unsqueeze(0)), 1)
                else:
                    s2 = torch.cat((s2, s_h2[i].unsqueeze(0)), 1)

            sa = s1.repeat(interpolate_num, 1, 1, 1)
            sb = s2.repeat(interpolate_num, 1, 1, 1)

            alpha = torch.linspace(start=0., end=1., steps=interpolate_num).unsqueeze(1)
            alpha = alpha.expand(sa.size(0), sa.nelement() // sa.size(0)).contiguous().view(sa.size())
            alpha = alpha.cuda()

            s_interpolates_ab = alpha * sb + ((1 - alpha) * sa)
            s_interpolates_ab = s_interpolates_ab.cuda()
            x_ab = self.gen.decode(ca, s_interpolates_ab)
            x_trans.append(x_ab[interpolate_num-1].unsqueeze(0))
            x_ab = torch.cat((x[i].unsqueeze(0), x_ab, x[perm_idx[i]].unsqueeze(0)))
            # comment it if you want to both save trans images and return them
            # if ret_trans:
            #     continue

            if h0+h1+h2==1:
                if h0:
                    lab_a = h0_label[i]
                    lab_b = h0_label[perm_idx[i]]
                    str_h = 'h0'
                elif h1:
                    lab_a = h1_label[i]
                    lab_b = h1_label[perm_idx[i]]
                    str_h = 'h1'
                elif h2:
                    lab_a = h2_label[i]
                    lab_b = h2_label[perm_idx[i]]
                    str_h = 'h2'

                _write_images(x_ab, '%s/%s_%03dTo%03d_%02dTo%02d.jpg' % (save_path, str_h, count+i, count+perm_idx[i],
                                                                         lab_a, lab_b), w_s, h_s)
            elif h0+h1+h2==2:
                if h0 and h1:
                    str_h ='h0h1'
                    lab_a1 = h0_label[i]
                    lab_a2 = h1_label[i]
                    lab_b1 = h0_label[perm_idx[i]]
                    lab_b2 = h1_label[perm_idx[i]]

                elif h0 and h2:
                    str_h = 'h0h2'
                    lab_a1 = h0_label[i]
                    lab_a2 = h2_label[i]
                    lab_b1 = h0_label[perm_idx[i]]
                    lab_b2 = h2_label[perm_idx[i]]
                elif h1 and h2:
                    str_h = 'h1h2'
                    lab_a1 = h1_label[i]
                    lab_a2 = h2_label[i]
                    lab_b1 = h1_label[perm_idx[i]]
                    lab_b2 = h2_label[perm_idx[i]]
                _write_images(x_ab, '%s/%s_%03dTo%03d_%02dTo%02d_%02dTo%02d.jpg' %
                              (save_path, str_h, count+i, count+perm_idx[i], lab_a1, lab_b1, lab_a2, lab_b2), w_s, h_s)

            else:
                str_h='h0h1h2'
                lab_a = h0_label[i]
                lab_b = h0_label[perm_idx[i]]
                _write_images(x_ab, '%s/%s_%03dTo%03d_%02dTo%02d.jpg' % (save_path, str_h, count+i, count+perm_idx[i],
                                                                          lab_a, lab_b), w_s, h_s)
        if ret_trans:
            return torch.cat(x_trans)

    # qss, h0, h1 labels needed, only one images data, both dis loss & auxiliary loss
    def dis_update(self, x, hyperparameters, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label=None):
        self.dis_opt.zero_grad()

        # encode
        c, s_prime = self.gen.encode(x)

        if self.root_num>1 and h2_label is not None:
            s_composite, _, _, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels, tar_h2_local_labels\
                = self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label, h2_local_label)
        else:
            s_composite, _, _, tar_h0_local_labels, tar_h1_labels, tar_h1_local_labels, tar_h2_labels = \
                self.permute_recompostion(s_prime, h0_local_label, h1_label, h1_local_label, h2_label)
            tar_h2_local_labels = None

        # decode (cross domain), cross generation only uses sampled style
        x_trans = self.gen.decode(c, s_composite)

        # D loss, detach for no grad for input fake images
        self.loss_dis_adv, self.loss_cls_h0, self.loss_cls_h1, self.loss_clc_h2 = \
            self.dis.calc_dis_loss(x_trans.detach(), x, h0_local_label, h1_label, h1_local_label, h2_label,
                                   h2_local_label)
        self.loss_dis = hyperparameters['gan_w'] * \
                        (self.loss_dis_adv + self.loss_cls_h0 + self.loss_cls_h1 + self.loss_clc_h2)
        self.loss_dis.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    # qss, gen, dis model
    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    # qss, gen, dis model
    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
