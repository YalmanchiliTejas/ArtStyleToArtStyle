from typing import Dict, Any, Optional
import torch
import  torch.nn.functional as F
import torch.nn as nn


def init_weights(module, gain=0.02):
    """
    Apply Kaiming-normal init to conv layers and constant to norms.
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(module.weight.data,0.0, gain)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1 or classname.find("BatchNorm2d") != -1:
        if module.weight is not None:
            torch.nn.init.normal_(module.weight.data, 1.0, gain)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)


class ResNet(torch.nn.Module):
    def __init__(self,dimensions, padding, norm_layer, dropout):
        super().__init__()

        if padding == 'reflect':
            self.padding = torch.nn.ReflectionPad2d
        elif padding == 'replicate':
            self.padding = torch.nn.ReplicationPad2d
        else:
            self.padding = torch.nn.ZeroPad2d
        layers = []
        layers += [self.padding(1),
                   torch.nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0,bias=False),
                   norm_layer(dimensions),
                   torch.nn.ReLU(True)]
        if dropout:
            layers.append(torch.nn.Dropout(0.5))
        layers += [self.padding(1),
                   torch.nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0,bias=False),
                   norm_layer(dimensions)]
        self.conv_block = torch.nn.Sequential(*layers)
    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResNetGenerator(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ngf, num_blocks, norm_layer):

        super().__init__()
        model = []
        model += [torch.nn.ReflectionPad2d(3), nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
                  norm_layer(ngf),
                  torch.nn.ReLU(True)]
        # Downsampling
        n_downsampling = 2
        mult = 1
        for _ in range(n_downsampling):
            model += [torch.nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                      stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2),
                      torch.nn.ReLU(True)]
            mult *= 2
        # ResNet blocks
        for _ in range(num_blocks):
            model += [ResNet(ngf * mult, 'reflect', norm_layer, False)]
        # Upsampling
        for _ in range(n_downsampling):
            model += [torch.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                               kernel_size=3, stride=2,
                                               padding=1, output_padding=1,
                                               bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      torch.nn.ReLU(True)]
            mult = mult // 2
        model += [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
                  torch.nn.Tanh()]
        self.model = torch.nn.Sequential(*model)
        self.apply(init_weights)
    def forward(self, input):
        return self.model(input)

class PatchGANDiscriminator(torch.nn.Module):
    def __init__(self, in_channels, ndf, n_layer, norm_layer):
        super().__init__()
        sequence = [torch.nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                    torch.nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layer):
            mult_prev = mult
            mult = min(2 ** n, 8)
            sequence += [torch.nn.Conv2d(ndf * mult_prev, ndf * mult,
                                         kernel_size=4, stride=2, padding=1, bias=False),
                         norm_layer(ndf * mult),
                         torch.nn.LeakyReLU(0.2, True)]
        mult_prev = mult
        mult = min(2 ** n_layer, 8)
        sequence += [torch.nn.Conv2d(ndf * mult_prev, ndf * mult,
                                     kernel_size=4, stride=1, padding=1, bias=False),
                        norm_layer(ndf * mult), torch.nn.LeakyReLU(0.2, True)]
        sequence += [torch.nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = torch.nn.Sequential(*sequence)
        self.apply(init_weights)
    def forward(self, input):
        return self.model(input)


class LSGANLoss(torch.nn.Module):
    def __init__(self, lsgan):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCEWithLogitsLoss()
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return self.real_label.expand_as(input)
        else:
            return self.fake_label.expand_as(input)
    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class BaselineCycleGan(torch.nn.Module):

    def __init__(self, in_channels, out_channels, lambda_cycle, lambda_identity):
        super().__init__()
        self.G_xy = ResNetGenerator(in_channels, out_channels, ngf=64, num_blocks=9, norm_layer=torch.nn.InstanceNorm2d)
        self.F_yx = ResNetGenerator(out_channels, in_channels, ngf=64, num_blocks=9, norm_layer=torch.nn.InstanceNorm2d)
        self.D_x = PatchGANDiscriminator(in_channels, ndf=64, n_layer=3, norm_layer=torch.nn.InstanceNorm2d)
        self.D_y = PatchGANDiscriminator(out_channels, ndf=64, n_layer=3, norm_layer=torch.nn.InstanceNorm2d)
        self.criterionGAN = LSGANLoss(lsgan=True)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdentity = torch.nn.L1Loss()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    def forward(self, real_x, real_y):

        # X to Y and back to X
        fake_y = self.G_xy(real_x)
        rec_x = self.F_yx(fake_y)

        # Y to X and back to Y
        fake_x = self.F_yx(real_y)
        rec_y = self.G_xy(fake_x)
        return real_x, real_y, fake_x, fake_y, rec_x, rec_y
    def compute_generator_loss(self, real_x, real_y):
        real_x, real_y, fake_x, fake_y, rec_x, rec_y = self.forward(real_x, real_y)

        pred_fake_y = self.D_y(fake_y)
        pred_fake_x = self.D_x(fake_x)
        # GAN loss D_y(G_xy(X))
        loss_G_xy = self.criterionGAN(pred_fake_y, True)
        # GAN loss D_x(F_yx(Y))
        loss_F_yx = self.criterionGAN(pred_fake_x, True)
        # Adversial loss
        loss_g_adversial = loss_G_xy + loss_F_yx
        # Cycle loss
        loss_cycle_x = self.criterionCycle(rec_x, real_x)
        loss_cycle_y = self.criterionCycle(rec_y, real_y)
        loss_cycle = loss_cycle_x + loss_cycle_y
        # Identity loss
        loss_id_x = self.criterionIdentity(self.F_yx(real_x), real_x)
        loss_id_y = self.criterionIdentity(self.G_xy(real_y), real_y)
        loss_identity = loss_id_x + loss_id_y
        # Total generator loss
        loss_G = loss_g_adversial + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity
        return {"loss_g_total": loss_G,
                "loss_g_adversial": loss_g_adversial,
                "loss_cycle": loss_cycle,
                "loss_identity": loss_identity}
    def compute_discriminator_loss(self, real_x, real_y):
        with torch.no_grad():
            fake_x = self.F_yx(real_y)
            fake_y = self.G_xy(real_x)
        # Discriminator D_x
        pred_real_x = self.D_x(real_x)
        pred_fake_x = self.D_x(fake_x)
        loss_D_x_real = self.criterionGAN(pred_real_x, True)
        loss_D_x_fake = self.criterionGAN(pred_fake_x, False)
        loss_D_x = (loss_D_x_real + loss_D_x_fake) * 0.5
        # Discriminator D_y
        pred_real_y = self.D_y(real_y)
        pred_fake_y = self.D_y(fake_y)
        loss_D_y_real = self.criterionGAN(pred_real_y, True)
        loss_D_y_fake = self.criterionGAN(pred_fake_y, False)
        loss_D_y = (loss_D_y_real + loss_D_y_fake) * 0.5
        # Total discriminator loss
        loss_D = loss_D_x + loss_D_y
        return {"loss_d_total": loss_D,
                "loss_d_x": loss_D_x,
                "loss_d_y": loss_D_y}


