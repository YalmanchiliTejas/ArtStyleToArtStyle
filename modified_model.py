
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------
#  Init + building blocks
# ---------------------------------------------------------

def init_weights(module, gain: float = 0.02):
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname or "BatchNorm2d" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


class ResNet(nn.Module):
    def __init__(self, dimensions, padding, norm_layer, dropout):
        super().__init__()

        if padding == "reflect":
            pad = nn.ReflectionPad2d
        elif padding == "replicate":
            pad = nn.ReplicationPad2d
        else:
            pad = nn.ZeroPad2d

        layers = [
            pad(1),
            nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dimensions),
            nn.ReLU(True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers += [
            pad(1),
            nn.Conv2d(dimensions, dimensions, kernel_size=3, stride=1, padding=0, bias=False),
            norm_layer(dimensions),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


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
            mult //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


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

    def forward(self, x, return_features: bool = False):
        if not return_features:
            return self.model(x)

        feats = []
        out = x
        for layer in self.model:
            out = layer(out)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.LeakyReLU):
                feats.append(out)
        return out, feats


# ---------------------------------------------------------
#  Hinge GAN loss (only)
# ---------------------------------------------------------

class HingeGANLoss(nn.Module):
    """
    Pure hinge loss:

      D_real:  E[ReLU(1 - D(x_real))]
      D_fake:  E[ReLU(1 + D(x_fake))]
      G:       -E[D(x_fake)]
    """

    def __init__(self):
        super().__init__()

    def forward(self, inp, target_is_real: bool, for_discriminator: bool = True):
        if for_discriminator:
            if target_is_real:
                return torch.mean(F.relu(1.0 - inp))
            else:
                return torch.mean(F.relu(1.0 + inp))
        # generator: want D(fake) to be large
        return -torch.mean(inp)


# ---------------------------------------------------------
#  VGG features + Gram
# ---------------------------------------------------------

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


# ---------------------------------------------------------
#  Improved CycleGAN (hinge only)
# ---------------------------------------------------------

class ImprovedCycleGan(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lambda_cycle,
        lambda_identity,
        lambda_content: float = 1.0,
        lambda_style: float = 1.0,
        lambda_fm: float = 1.0,
        gan_mode: str = "hinge",
    ):
        super().__init__()
        assert gan_mode.lower() == "hinge", "ImprovedCycleGan is hinge-only in this version."

        # Generators
        self.G_xy = ResNetGenerator(
            in_channels, out_channels, ngf=64, num_blocks=9, norm_layer=nn.InstanceNorm2d
        )
        self.F_yx = ResNetGenerator(
            out_channels, in_channels, ngf=64, num_blocks=9, norm_layer=nn.InstanceNorm2d
        )

        # Discriminators
        self.D_x = PatchGANDiscriminator(
            in_channels, ndf=64, n_layer=3, norm_layer=nn.InstanceNorm2d
        )
        self.D_y = PatchGANDiscriminator(
            out_channels, ndf=64, n_layer=3, norm_layer=nn.InstanceNorm2d
        )

        # Loss helpers
        self.criterionGAN = HingeGANLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdentity = nn.L1Loss()

        # VGG
        self.vgg = VGG19Features(requires_grad=False)

        # Weights
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_content = lambda_content
        self.lambda_style = lambda_style
        self.lambda_fm = lambda_fm

    # ---------- helpers ----------

    def _vgg_feats(self, img: torch.Tensor):
        # input in [-1,1] -> [0,1]
        img_norm = (img + 1.0) / 2.0
        return self.vgg(img_norm)

    # ---------- forward ----------

    def forward(self, real_x, real_y):
        # X -> Y -> X
        fake_y = self.G_xy(real_x)
        rec_x = self.F_yx(fake_y)

        # Y -> X -> Y
        fake_x = self.F_yx(real_y)
        rec_y = self.G_xy(fake_x)

        return real_x, real_y, fake_x, fake_y, rec_x, rec_y

    # ---------- generator loss ----------

    def compute_generator_loss(self, real_x, real_y) -> Dict[str, torch.Tensor]:
        real_x, real_y, fake_x, fake_y, rec_x, rec_y = self.forward(real_x, real_y)

        # Adversarial
        pred_fake_y = self.D_y(fake_y)
        pred_fake_x = self.D_x(fake_x)
        loss_G_xy = self.criterionGAN(pred_fake_y, True, for_discriminator=False)
        loss_F_yx = self.criterionGAN(pred_fake_x, True, for_discriminator=False)
        loss_g_adversial = loss_G_xy + loss_F_yx

        # Cycle
        loss_cycle_x = self.criterionCycle(rec_x, real_x)
        loss_cycle_y = self.criterionCycle(rec_y, real_y)
        loss_cycle = loss_cycle_x + loss_cycle_y

        # Identity
        loss_id_x = self.criterionIdentity(self.F_yx(real_x), real_x)
        loss_id_y = self.criterionIdentity(self.G_xy(real_y), real_y)
        loss_identity = loss_id_x + loss_id_y

        # Perceptual + Style (using VGG)
        feats_real_x = self._vgg_feats(real_x)
        feats_real_y = self._vgg_feats(real_y)
        feats_fake_x = self._vgg_feats(fake_x)
        feats_fake_y = self._vgg_feats(fake_y)

        # Content: preserve input structure
        content_loss_xy = F.l1_loss(feats_fake_y[3], feats_real_x[3])  # X->Y: content from X
        content_loss_yx = F.l1_loss(feats_fake_x[3], feats_real_y[3])  # Y->X: content from Y
        loss_content = self.lambda_content * (content_loss_xy + content_loss_yx)

        # Style: match target-domain style
        style_loss_xy = 0.0
        style_loss_yx = 0.0
        for f_fy, f_y in zip(feats_fake_y, feats_real_y):
            style_loss_xy = style_loss_xy + F.l1_loss(gram_matrix(f_fy), gram_matrix(f_y))
        for f_fx, f_x in zip(feats_fake_x, feats_real_x):
            style_loss_yx = style_loss_yx + F.l1_loss(gram_matrix(f_fx), gram_matrix(f_x))
        loss_style = self.lambda_style * (style_loss_xy + style_loss_yx)

        # Feature matching
        fm_raw = 0.0
        _, feats_real_x_D = self.D_x(real_x, return_features=True)
        _, feats_fake_x_D = self.D_x(fake_x, return_features=True)
        for fr, ff in zip(feats_real_x_D, feats_fake_x_D):
            fm_raw = fm_raw + F.l1_loss(ff, fr.detach())

        _, feats_real_y_D = self.D_y(real_y, return_features=True)
        _, feats_fake_y_D = self.D_y(fake_y, return_features=True)
        for fr, ff in zip(feats_real_y_D, feats_fake_y_D):
            fm_raw = fm_raw + F.l1_loss(ff, fr.detach())

        loss_fm = self.lambda_fm * fm_raw

        # Total G
        loss_G = (
            loss_g_adversial
            + self.lambda_cycle * loss_cycle
            + self.lambda_identity * loss_identity
            + loss_content
            + loss_style
            + loss_fm
        )

        return {
            "loss_g_total": loss_G,
            "loss_g_adversial": loss_g_adversial,
            "loss_cycle": loss_cycle,
            "loss_identity": loss_identity,
            "loss_content": loss_content,
            "loss_style": loss_style,
            "loss_fm": loss_fm,
        }

    # ---------- discriminator loss ----------

    def compute_discriminator_loss(self, real_x, real_y) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            fake_x = self.F_yx(real_y)
            fake_y = self.G_xy(real_x)

        # D_x
        pred_real_x = self.D_x(real_x)
        pred_fake_x = self.D_x(fake_x)
        loss_D_x_real = self.criterionGAN(pred_real_x, True, for_discriminator=True)
        loss_D_x_fake = self.criterionGAN(pred_fake_x, False, for_discriminator=True)
        loss_D_x = 0.5 * (loss_D_x_real + loss_D_x_fake)

        # D_y
        pred_real_y = self.D_y(real_y)
        pred_fake_y = self.D_y(fake_y)
        loss_D_y_real = self.criterionGAN(pred_real_y, True, for_discriminator=True)
        loss_D_y_fake = self.criterionGAN(pred_fake_y, False, for_discriminator=True)
        loss_D_y = 0.5 * (loss_D_y_real + loss_D_y_fake)

        loss_D = loss_D_x + loss_D_y

        return {
            "loss_d_total": loss_D,
            "loss_d_x": loss_D_x,
            "loss_d_y": loss_D_y,
        }

