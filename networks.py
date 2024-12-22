from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

# Generator- AdaINGen, call Decoder, Encoder, sequential blocks
# Multi-scale discriminator- MsImageDis, call basic blocks

# Encoder- StyleEncoder, ContentEncoder, call sequential and basic blocks
# Decoder, call sequential and basic blocks

# Sequential- ResBlocks, MLP modules, call basic blocks
# Basic blocks- ResBlock, Conv2dBlock, LinearBlock, AdaptiveInstanceNorm2d, LayerNorm


##################################################################################
# Discriminator
##################################################################################

# qss auxiliary classifier
class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tree = params['tree']
        self.h2_classes = self.tree[self.tree[0] + 1]
        self.h1_classes = self.tree[0]
        self.h0_classes = sum(self.tree[1:self.tree[0] + 1])
        self.cnns = nn.ModuleList()
        self.aux_h0 = nn.ModuleList()
        self.aux_h1 = nn.ModuleList()
        if self.h2_classes > 1:
            self.aux_h2 = nn.ModuleList()
        self.dis = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())
            self.aux_h0.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), self.h0_classes, 1, 1, 0))
            self.aux_h1.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), self.h1_classes, 1, 1, 0))
            self.dis.append(nn.Conv2d(self.dim * (2 ** (self.n_layer-1)), 1, 1, 1, 0))
            if self.h2_classes > 1:
                self.aux_h2.append(nn.Conv2d(self.dim*(2**(self.n_layer-1)), self.h2_classes, 1, 1, 0))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        if self.norm == 'sn':
            cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
        else:
            cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ,
                                  pad_type=self.pad_type)]

        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs_dis, outputs_aux_h0, outputs_aux_h1, outputs_aux_h2 = [], [], [], []
        for it, (cnn, cls_h0, cls_h1, dis) in enumerate(zip(self.cnns, self.aux_h0, self.aux_h1, self.dis)):
            tmp = cnn(x)
            tmp_h0 = cls_h0(tmp)
            tmp_h0 = self.global_avg_pool(tmp_h0).squeeze()
            tmp_h1 = cls_h1(tmp)
            tmp_h1 = self.global_avg_pool(tmp_h1).squeeze()
            tmp_dis = dis(tmp)
            outputs_aux_h0.append(tmp_h0)
            outputs_aux_h1.append(tmp_h1)
            outputs_dis.append(tmp_dis)
            if self.h2_classes > 1:
                tmp_h2 = self.aux_h2[it](tmp)
                tmp_h2 = self.global_avg_pool(tmp_h2).squeeze()
                outputs_aux_h2.append(tmp_h2)
            x = self.downsample(x)
        return outputs_dis, outputs_aux_h0, outputs_aux_h1, outputs_aux_h2

    def calc_dis_loss(self, input_fake, input_real, h0_local_labels, h1_labels, h1_local_labels, h2_labels,
                      h2_local_labels=None):
        # calculate the loss to train D
        outs0, outs0_h0, outs0_h1, outs0_h2 = self.forward(input_fake)
        outs1, outs1_h0, outs1_h1, outs1_h2 = self.forward(input_real)
        loss_adv, loss_h0, loss_h1, loss_h2 = 0., 0., 0., 0.

        alpha = torch.rand(input_real.size(0), 1)
        alpha = alpha.expand(input_real.size(0), input_real.nelement() // input_real.size(0)).contiguous().\
            view(input_real.size())
        alpha = alpha.cuda()

        interpolates = alpha * input_real + ((1 - alpha) * input_fake)
        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        disc_interpolates, _, _, _ = self.forward(interpolates)

        for it, (out0, out1, out_inter) in enumerate(zip(outs0, outs1, disc_interpolates)):
            if self.gan_type == 'lsgan':
                loss_adv += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':  # original gan ?
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                       F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'wgan-gp':
                loss_adv += torch.mean(out0) - torch.mean(out1)
                gradients = torch.autograd.grad(outputs=out_inter, inputs=interpolates,
                                                grad_outputs=torch.ones(out_inter.size()).cuda(),
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradients = gradients.view(gradients.size(0), -1)
                loss_adv += ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss_h0 = self.calc_auxiliary_loss_h0(outs1_h0, h0_local_labels, h1_labels)

        loss_h1 = self.calc_auxiliary_loss_h1(outs1_h1, h1_local_labels, h2_labels)

        if self.h2_classes > 1:
            loss_h2 = self.calc_auxiliary_loss_h2(outs1_h2, h2_local_labels)

        return loss_adv, loss_h0, loss_h1, loss_h2

    def calc_gen_loss(self, input_fake, h0_local_labels, h1_labels, h1_local_labels, h2_labels, h2_local_labels=None):
        # calculate the loss to train G
        outs0, outs_h0, outs_h1, outs_h2 = self.forward(input_fake)
        loss_adv, loss_aux_h0, loss_aux_h1, loss_aux_h2 = 0., 0., 0., 0.
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss_adv += torch.mean((out0 - 1)**2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss_adv += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'wgan-gp':
                loss_adv += -torch.mean(out0)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        loss_aux_h0 = self.calc_auxiliary_loss_h0(outs_h0, h0_local_labels.cuda(), h1_labels.cuda())
        loss_aux_h1 = self.calc_auxiliary_loss_h1(outs_h1, h1_local_labels.cuda(), h2_labels.cuda())
        if self.h2_classes > 1:
            loss_aux_h2 = self.calc_auxiliary_loss_h2(outs_h2, h2_local_labels.cuda())

        return loss_adv, loss_aux_h0, loss_aux_h1, loss_aux_h2

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

    def calc_auxiliary_loss_h0(self, outs_h0, h0_local_labels, h1_labels):
        loss = 0
        for it, (out_h0) in enumerate(outs_h0):
            loss += self.compute_local_cls_loss(out_h0, self.tree[1:self.tree[0]+1], self.h1_classes, h0_local_labels,
                                                h1_labels)
        return loss

    def calc_auxiliary_loss_h1(self, outs_h1, h1_local_labels, h2_labels):
        loss = 0
        for it, (out_h1) in enumerate(outs_h1):
            if self.h2_classes == 1:
                loss += torch.mean(F.cross_entropy(out_h1, h1_local_labels))
            else:
                loss += self.compute_local_cls_loss(out_h1, self.tree[self.tree[0]+2:], self.h2_classes,
                                                    h1_local_labels, h2_labels)
        return loss

    def calc_auxiliary_loss_h2(self, outs_h2, h2_labels):
        loss = 0
        for it, (out_h2) in enumerate(outs_h2):
            loss += torch.mean(F.cross_entropy(out_h2, h2_labels))
        return loss


##################################################################################
# Generator
##################################################################################
class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        self.tree = params['tree']
        self.root_num = self.tree[self.tree[0]+1]
        self.intermediate_num = self.tree[0]
        self.leaves_num = sum(self.tree[1:self.tree[0] + 1])

        # style encoder, qss
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type,
                                      root_num=self.root_num)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ,
                           pad_type=pad_type)

        if self.root_num>1:
            fea_dim = style_dim*3
        else:
            fea_dim = style_dim*2

        # MLP to generate AdaIN parameters
        self.mlp = MLP(fea_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

        # hierarchical classifiers
        self.cls_h0 = LinearBlock(fea_dim, self.leaves_num, norm='none', activation='none')
        self.cls_h1 = LinearBlock(fea_dim, self.intermediate_num, norm='none', activation='none')
        if self.root_num > 1:
            self.cls_h2 = LinearBlock(fea_dim, self.root_num, norm='none', activation='none')

    def forward(self, images):
        # reconstruct an image
        content, style_hier = self.encode(images)
        style = torch.cat(style_hier, 1)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_hier = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_hier

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


##################################################################################
# Encoder and Decoders
##################################################################################

# qss, style dim
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type, root_num):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model = nn.Sequential(*self.model)
        self.model_h0 = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.model_h1 = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.root_num = root_num
        if root_num > 1:
            self.model_h2 = nn.Conv2d(dim, style_dim, 1, 1, 0)
        self.output_dim = style_dim

    def forward(self, x):
        x = self.model(x)
        if self.root_num > 1:
            return self.model_h0(x), self.model_h1(x), self.model_h2(x)
        else:
            return self.model_h0(x), self.model_h1(x)


# qss, n_downsample
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


# qss, n_upsample
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks, qss use layer norm ?
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer, binary output
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out  # no relu for output


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):  # padding: size to pad not value
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x.view(x.size(0), -1))
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers, needs weight and bias be assigned before called
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned, i.e. gama and beta ?
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

