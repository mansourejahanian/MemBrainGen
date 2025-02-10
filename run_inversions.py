import os
import argparse

import dill
import h5py
from stylegan_xl import dnnlib
import numpy as np
import pandas as pd
import PIL.Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models
import timm

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import seaborn as sns
from scipy import stats
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from stylegan_xl import legacy
from stylegan_xl.torch_utils import gen_utils
from stylegan_xl.inversion_utils import project, pivotal_tuning


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
    # centroids_path = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet_centroids.npy"
    device = torch.device(f'cuda:{args.device_id}')

    # load pretrained StyleGAN-XL
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.eval().requires_grad_(False).to(device)

    print('Loaded networks from "%s"' % network_pkl)

    data_dir = "data/nsd/"
    save_dir = "/home/hhan228/memorability/Willow/shared1000_inversions/"
    img_dir = data_dir+"shared1000/"

    target_images = sorted([f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))])

    inv_steps = 1000
    pti_steps = 350

    for img_idx in range(args.start_idx, args.end_idx):
        print(f"shared{img_idx+1:04d}")
        img_save_dir = save_dir+f"shared{img_idx+1:04d}/"
        os.makedirs(img_save_dir, exist_ok=True)

        target_pil = PIL.Image.open(img_dir+target_images[img_idx]).convert("RGB").resize((256, 256))

        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)

        all_images, projected_w, imagenet_class = project(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            num_steps=inv_steps,
            device=device,
            verbose=False,
            noise_mode='const',
        )

        gen_images, G = pivotal_tuning(
            G,
            projected_w,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
            device=device,
            num_steps=pti_steps,
            verbose=False,
        )
        all_images += gen_images

        target_pil.save(img_save_dir+'target.png')
        synth_image = G.synthesis(projected_w.repeat(1, G.num_ws, 1))
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        proj_img = PIL.Image.fromarray(synth_image, 'RGB')
        proj_img.save(img_save_dir+'proj.png')
        np.save(img_save_dir+'projected_w.npy', projected_w.repeat(1, G.num_ws, 1).cpu().numpy())

        snapshot_data = {'G_ema': G}
        with open(img_save_dir+"G.pkl", 'wb') as f:
            dill.dump(snapshot_data, f)