import os
import cv2
import numpy as np
from pathlib import Path
import torch
from torch import nn as nn
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ColorJitter, RandomResizedCrop, ToTensor, Compose, ToPILImage
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

transforms = Compose([
    ColorJitter(),
    #RandomResizedCrop(size=(256, 256), ratio=(1.0, 1.0)),
    ToTensor()
])

GEN_DOWNSCALER_CHANNELS = [3, 64, 128, 256, 512, 512, 512, 512, 512]
DIS_DOWNSCALER_CHANNELS = [6, 64, 128, 256, 512, 512, 512, 512, 512]

CHANNELS_IN_GEN = GEN_DOWNSCALER_CHANNELS[:-1]
CHANNELS_OUT_GEN = GEN_DOWNSCALER_CHANNELS[1:]

CHANNELS_OUT_DIS = DIS_DOWNSCALER_CHANNELS[:-1]

SPATIAL_SIZE_D_DOWN = [256, 128, 64, 32, 16, 8, 4, 2, 1]
SPATIAL_SIZE_D_UP = list(reversed(SPATIAL_SIZE_D_DOWN))

CHANNELS_IN_DIS = [512] + [
    512 * 2, 512 * 2, 512 * 2, 512 * 2, 256 * 2, 128 * 2, 64 * 2
]
D_CHANNELS_OUT = [512, 512, 512, 512, 256, 128, 64, 3]


def get_rgb_image(file_name):
    try:
        img = cv2.imread(file_name)
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        print(f"Failed to read image: ", file_name)
    return img


def cv2np(img):
    img = np.asarray(img)
    assert img.shape[2] == 3
    return img


def np2pil(img):
    return Image.fromarray(img)


def img2pil(file_name):
    img = get_rgb_image(file_name)
    img = np2pil(cv2np(img))
    return img


class Image2ImageDataset(torch.utils.data.Dataset):
    """
    Dataset for the Faceades dataset (base + extended)
    """
    def __init__(self, root_dir):
        self.root_dir = os.path.join(root_dir)
        self.image_names = [
            str(x.split(".")[0]) for x in os.listdir(self.root_dir)
            if x.endswith(".jpg")
        ]

    def __len__(self):
        return len(list(map(str, Path(self.root_dir).glob("*.jpg"))))

    def __getitem__(self, idx):
        images = [
            os.path.join(self.root_dir, self.image_names[idx]) + ".jpg",
            os.path.join(self.root_dir, self.image_names[idx]) + ".png"
        ]
        images = list(map(img2pil, images))
        images = [transforms(img) for img in images]
        return images


class DownScaleBlock(nn.Module):
    def __init__(self, c_idx, disc=False):
        super(DownScaleBlock, self).__init__()
        self.same_padding = (((SPATIAL_SIZE_D_DOWN[c_idx + 1] - 1) * 2) + 4 -
                             SPATIAL_SIZE_D_DOWN[c_idx]) // 2
        self.block = []
        if disc:
            conv = nn.Conv2d(in_channels=CHANNELS_OUT_DIS[c_idx],
                             out_channels=CHANNELS_OUT_GEN[c_idx],
                             kernel_size=4,
                             padding=self.same_padding,
                             stride=2)
        else:
            conv = nn.Conv2d(in_channels=CHANNELS_IN_GEN[c_idx],
                             out_channels=CHANNELS_OUT_GEN[c_idx],
                             kernel_size=4,
                             padding=self.same_padding,
                             stride=2)
        bn = nn.BatchNorm2d(num_features=CHANNELS_OUT_GEN[c_idx])
        act = nn.LeakyReLU()

        self.block.append(conv)
        if c_idx != 0 and c_idx != 7:
            self.block.append(bn)
        self.block.append(act)
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class UpScaleBlock(nn.Module):
    def __init__(self, c_idx):
        super(UpScaleBlock, self).__init__()
        self.idx = c_idx
        self.block = []

        conv = nn.ConvTranspose2d(in_channels=CHANNELS_IN_DIS[c_idx],
                                  out_channels=D_CHANNELS_OUT[c_idx],
                                  kernel_size=4,
                                  stride=2,
                                  padding=1)
        self.block.append(conv)

        if c_idx != 0 and c_idx != 7:
            bn = nn.BatchNorm2d(num_features=D_CHANNELS_OUT[c_idx])
            self.block.append(bn)

        if c_idx != 7:
            act = nn.ReLU()
            self.block.append(act)

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downscaler = nn.ModuleList([
            DownScaleBlock(idx)
            for idx in range(len(GEN_DOWNSCALER_CHANNELS) - 1)
        ]).cuda()
        self.upscaler = nn.ModuleList([
            UpScaleBlock(idx)
            for idx in range(len(GEN_DOWNSCALER_CHANNELS) - 1)
        ]).cuda().cuda()

    def forward(self, x):
        skips = []
        for layer_d in self.downscaler:
            x = layer_d(x)
            skips.append(x)

        skips = list(reversed(skips))[1:]

        for idx, (layer_u, _skip) in enumerate(list(zip(self.upscaler,
                                                        skips))):
            x = layer_u(x)
            x = torch.cat((x, _skip), dim=1)

        x = self.upscaler[-1](x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downscaler = [DownScaleBlock(idx, True) for idx in range(3)]
        self.blocks = nn.Sequential(*self.downscaler).cuda()
        self.patch1 = nn.Sequential(nn.Conv2d(256, 256, 4, 1, padding=1),
                                    nn.BatchNorm2d(256), nn.ReLU()).cuda()
        self.patch2 = nn.Sequential(nn.Conv2d(256, 1, 4, 1, padding=1),
                                    nn.Sigmoid()).cuda()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.blocks(x)
        x = self.patch1(x)
        x = self.patch2(x)
        return x


if __name__ == "__main__":
    gen = Generator().train().cuda()
    dis = Discriminator().train().cuda()

    ds = Image2ImageDataset(
        root_dir="/home/rex/datasets/CMP_facade/base_plus_extended")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    bce_criteria = nn.BCELoss().cuda()
    l1_criteria = nn.L1Loss().cuda()

    optim_d = torch.optim.Adam(dis.parameters(), lr=2e-4)
    optim_g = torch.optim.Adam(gen.parameters(), lr=2e-4)

    global_step = 0
    for epoch in range(500):
        for idx, data in enumerate(dataloader):
            target_image, input_image = data
            target_image = target_image.to(torch.device('cuda'))
            input_image = input_image.to(torch.device('cuda'))

            optim_d.zero_grad()
            gen_op = gen(input_image)
            disc_real = dis(input_image, target_image)
            disc_fake = dis(input_image, gen_op)
            disc_loss = bce_criteria(
                disc_real, Variable(
                    torch.ones_like(disc_real))) + bce_criteria(
                        disc_fake, Variable(torch.zeros_like(disc_fake))) * 0.5
            disc_loss.backward()
            optim_d.step()
            writer.add_scalar('loss/discriminator', disc_loss, global_step=idx)

            optim_g.zero_grad()
            gen_op = gen(input_image)
            disc_fake = dis(input_image, gen_op)
            gen_loss = bce_criteria(
                disc_fake, Variable(torch.ones_like(disc_fake))) + l1_criteria(
                    input_image, gen_op)
            gen_loss.backward()
            optim_g.step()
            writer.add_scalar('loss/generator', gen_loss, global_step=idx)

            if idx % 50 == 0:
                writer.add_images('images/input_images',
                                  input_image,
                                  global_step=global_step)
                writer.add_images('images/target_images',
                                  target_image,
                                  global_step=global_step)
                writer.add_images('images/generated_images',
                                  gen_op.detach(),
                                  global_step=global_step)

                if not os.path.exists(f"../data/{str(idx)}"):
                    os.makedirs(f"../data/{str(idx)}")
                img = gen_op.cpu()[0]
                img = np.array(ToPILImage()(img))
                cv2.imwrite(f"../data/{str(idx)}/e_{str(epoch)}.jpg",
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                torch.save({
                    'gen': gen.state_dict(),
                    'disc': dis.state_dict(),
                    'optim_D': optim_d.state_dict(),
                    'optim_G': optim_g.state_dict(),
                    'epoch': epoch
                }, f"./models/e{str(epoch)}.pth")
                print(
                    f"epoch: {epoch}, idx: {idx}, Generator loss: {gen_loss.detach()} discriminator loss: {disc_loss.detach()}"
                )
            global_step += 1
