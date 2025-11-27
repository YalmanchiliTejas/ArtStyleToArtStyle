from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def init_weights(module, gain=0.02):
    """
    Apply Kaiming-normal init to conv layers and constant to norms.
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1 or classname.find("BatchNorm2d") != -1:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(self, dimensions, padding, norm_layer, dropout):
        super().__init__()

        if padding == "reflect":
            self.padding = nn.ReflectionPad2d
        elif padding == "replicate":
            self.padding = nn.ReplicationPad2d
        else:
            self.padding = nn.ZeroPad2d

        layers = [
            self.padding(1),
            nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dimensions),
            nn.ReLU(True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            self.padding(1),
            nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dimensions),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, num_blocks, norm_layer):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        # Downsampling
        n_downsampling = 2
        mult = 1
        for _ in range(n_downsampling):
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2
        # ResNet blocks
        for _ in range(num_blocks):
            model += [ResNet(ngf * mult, "reflect", norm_layer, False)]
        # Upsampling
        for _ in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
            mult = mult // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
        self.apply(init_weights)

    def forward(self, input):
        return self.model(input)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf, n_layer, norm_layer):
        super().__init__()
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        mult = 1
        for n in range(1, n_layer):
            mult_prev = mult
            mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * mult_prev,
                    ndf * mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
        mult_prev = mult
        mult = min(2**n_layer, 8)
        sequence += [
            nn.Conv2d(
                ndf * mult_prev,
                ndf * mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)
        self.apply(init_weights)

    def forward(self, input, return_features: bool = False):
        if not return_features:
            return self.model(input)

        features = []
        out = input
        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LeakyReLU):
                features.append(out)
        return out, features


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str = "lsgan"):
        super().__init__()
        gan_mode = gan_mode.lower()
        assert gan_mode in ["lsgan", "vanilla", "hinge"]
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = None

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input)
        else:
            return torch.zeros_like(input)

    def forward(self, input, target_is_real, for_discriminator: bool = True):
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)

        if for_discriminator:
            if target_is_real:
                return torch.mean(F.relu(1.0 - input))
            else:
                return torch.mean(F.relu(1.0 + input))
        else:
            return -torch.mean(input)


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.size()
    feat = feat.view(b, c, -1)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (c * h * w)


class VGG19Features(nn.Module):
    def __init__(self, requires_grad: bool = False):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(12, 21)])
        self.slice5 = nn.Sequential(*[vgg[x] for x in range(21, 30)])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        relu1_1 = h
        h = self.slice2(h)
        relu2_1 = h
        h = self.slice3(h)
        relu3_1 = h
        h = self.slice4(h)
        relu4_1 = h
        h = self.slice5(h)
        relu5_1 = h
        return [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]


class PerceptualStyleLoss(nn.Module):
    def __init__(self, content_weight: float = 1.0, style_weight: float = 1.0):
        super().__init__()
        self.vgg = VGG19Features(requires_grad=False)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.criterion = nn.L1Loss()

    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor):
        input_norm = (input_img + 1.0) / 2.0
        target_norm = (target_img + 1.0) / 2.0

        input_feats = self.vgg(input_norm)
        with torch.no_grad():
            target_feats = self.vgg(target_norm)

        content_loss = self.criterion(input_feats[3], target_feats[3])

        style_loss = 0.0
        for f_in, f_tgt in zip(input_feats, target_feats):
            style_loss = style_loss + self.criterion(gram_matrix(f_in), gram_matrix(f_tgt))

        total_content = self.content_weight * content_loss
        total_style = self.style_weight * style_loss
        return total_content, total_style


class BaselineCycleGan(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lambda_cycle,
        lambda_identity,
        lambda_content: float = 1.0,
        lambda_style: float = 1.0,
        lambda_fm: float = 1.0,
        gan_mode: str = "lsgan",
    ):
        super().__init__()
        self.G_xy = ResNetGenerator(
            in_channels, out_channels, ngf=64, num_blocks=9, norm_layer=nn.InstanceNorm2d
        )
        self.F_yx = ResNetGenerator(
            out_channels, in_channels, ngf=64, num_blocks=9, norm_layer=nn.InstanceNorm2d
        )
        self.D_x = PatchGANDiscriminator(
            in_channels, ndf=64, n_layer=3, norm_layer=nn.InstanceNorm2d
        )
        self.D_y = PatchGANDiscriminator(
            out_channels, ndf=64, n_layer=3, norm_layer=nn.InstanceNorm2d
        )
        self.criterionGAN = GANLoss(gan_mode=gan_mode)
        self.criterionCycle = nn.L1Loss()
        self.criterionIdentity = nn.L1Loss()
        self.perceptual_style_loss = PerceptualStyleLoss(
            content_weight=lambda_content, style_weight=lambda_style
        )
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_fm = lambda_fm

    def forward(self, real_x, real_y):
        # X to Y and back to X
        fake_y = self.G_xy(real_x)
        rec_x = self.F_yx(fake_y)

        # Y to X and back to Y
        fake_x = self.F_yx(real_y)
        rec_y = self.G_xy(fake_x)
        return real_x, real_y, fake_x, fake_y, rec_x, rec_y

    def compute_generator_loss(self, real_x, real_y) -> Dict[str, torch.Tensor]:
        real_x, real_y, fake_x, fake_y, rec_x, rec_y = self.forward(real_x, real_y)

        pred_fake_y = self.D_y(fake_y)
        pred_fake_x = self.D_x(fake_x)
        # GAN loss D_y(G_xy(X))
        loss_G_xy = self.criterionGAN(pred_fake_y, True, for_discriminator=False)
        # GAN loss D_x(F_yx(Y))
        loss_F_yx = self.criterionGAN(pred_fake_x, True, for_discriminator=False)
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

        # Perceptual (content) loss
        content_loss_xy, style_loss_xy = self.perceptual_style_loss(fake_y, real_y)
        content_loss_yx, style_loss_yx = self.perceptual_style_loss(fake_x, real_x)
        loss_content = content_loss_xy + content_loss_yx
        # Style loss
        loss_style = style_loss_xy + style_loss_yx

        # Feature matching loss
        fm_loss = 0.0
        _, feats_real_x = self.D_x(real_x, return_features=True)
        _, feats_fake_x = self.D_x(fake_x, return_features=True)
        for fr, ff in zip(feats_real_x, feats_fake_x):
            fm_loss = fm_loss + F.l1_loss(ff, fr.detach())
        _, feats_real_y = self.D_y(real_y, return_features=True)
        _, feats_fake_y = self.D_y(fake_y, return_features=True)
        for fr, ff in zip(feats_real_y, feats_fake_y):
            fm_loss = fm_loss + F.l1_loss(ff, fr.detach())

        # Total generator loss
        loss_G = (
            loss_g_adversial
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
            + loss_content
            + loss_style
            + self.lambda_fm * fm_loss
        )
        return {
            "loss_g_total": loss_G,
            "loss_g_adversial": loss_g_adversial,
            "loss_cycle": loss_cycle,
            "loss_identity": loss_identity,
            "loss_content": loss_content,
            "loss_style": loss_style,
            "loss_fm": fm_loss,
        }

    def compute_discriminator_loss(self, real_x, real_y) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            fake_x = self.F_yx(real_y)
            fake_y = self.G_xy(real_x)
        # Discriminator D_x
        pred_real_x = self.D_x(real_x)
        pred_fake_x = self.D_x(fake_x)
        loss_D_x_real = self.criterionGAN(pred_real_x, True, for_discriminator=True)
        loss_D_x_fake = self.criterionGAN(pred_fake_x, False, for_discriminator=True)
        loss_D_x = (loss_D_x_real + loss_D_x_fake) * 0.5
        # Discriminator D_y
        pred_real_y = self.D_y(real_y)
        pred_fake_y = self.D_y(fake_y)
        loss_D_y_real = self.criterionGAN(pred_real_y, True, for_discriminator=True)
        loss_D_y_fake = self.criterionGAN(pred_fake_y, False, for_discriminator=True)
        loss_D_y = (loss_D_y_real + loss_D_y_fake) * 0.5
        # Total discriminator loss
        loss_D = loss_D_x + loss_D_y
        return {
            "loss_d_total": loss_D,
            "loss_d_x": loss_D_x,
            "loss_d_y": loss_D_y,
        }
