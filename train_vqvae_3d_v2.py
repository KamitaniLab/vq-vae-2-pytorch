import argparse
import sys
import os
import shutil
import pathlib
import pickle
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image

from torchvision import datasets, transforms

from tqdm import tqdm

from vqvae_3d_v2 import VQVAE
from scheduler import CycleScheduler
import distributed as dist

sys.path.append('/home/shirakawa/kspy')
from utils import load_video, get_cnn_features, save_video, save_gif

sys.path.append('/home/shirakawa/kspy/torch_videovision')
from videotransforms import video_transforms, volume_transforms, tensor_transforms

class RandomSelectFrames(object):
    """Crop a list of (H x W x C) np.array into targetd length
    """

    def __init__(self, length=16):
        self.length= length

    def __call__(self, clip):
        frame_num = clip.shape[0]

        if frame_num - self.length <=0:
            f_s = 0
        else:
            f_s = np.random.randint(0, frame_num-self.length)
        return clip[f_s:f_s+16]

class MITDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        #load video
        vid_path = self.file_list[index]
        vid = load_video(vid_path, 'float')
        vid = vid.astype(np.float32)
        #print(vid.shape)
        vid_transformed = self.transform(vid)

        return vid_transformed, vid_path

def save_videos(
    tensor: Union[torch.tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    #print(2)
    os.makedirs('sample_frame_v2', exist_ok=True)
    s_size, channel, fr, h, w = tensor.shape
    f_name_base, fmt = fp.split('.')
    for f in range(fr):
        tensor_tmp = tensor[:,:,f]
        f_name = '-'.join([f_name_base, f'fr_{f}', f'.{fmt}'])
        #print(f_name)
        save_image(tensor_tmp,
                   f_name, **kwargs)
        
    merge_list = []
    for f in range(fr):
        f_name = '-'.join([f_name_base, f'fr_{f}', f'.{fmt}'])
        ee = Image.open(f_name)
        merge_list.append(np.array(ee))

    merge_list = np.array(merge_list)
    
    f_name_base =f_name_base.replace('sample_frame_v2', 'sample_video_v2')
    os.makedirs('sample_video_v2', exist_ok=True)
    save_name = f_name_base + '.avi'


    save_video(merge_list,save_name, '.', bgr=False, fr_rate=16)
    save_name = f_name_base +'.gif'
    save_gif(merge_list,save_name, '.', bgr=False, fr_rate=60)   

    #remove current  frame files 
    shutil.rmtree('sample_frame_v2')
     

def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                save_videos(
                    torch.cat([sample, out], 0),
                    f"sample_frame_v2/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transforms = video_transforms.Compose(
    [   RandomSelectFrames(16),
        video_transforms.Resize(args.size),
        video_transforms.CenterCrop(args.size),
        volume_transforms.ClipToTensor(),
        tensor_transforms.Normalize(0.5,0.5)
    ])

    f = open('/home/shirakawa/movie/code/iVideoGAN/over16frame_list_training.txt', 'rb')
    train_file_list = pickle.load(f)
    print(len(train_file_list))

    dataset = MITDataset(train_file_list,transform= transforms)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    #loader = DataLoader(
    #    dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    #)
    loader = DataLoader(
        dataset, batch_size=32 // args.n_gpu, sampler=sampler, num_workers=2
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint_vid_v2/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    #parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
