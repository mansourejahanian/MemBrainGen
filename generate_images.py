import os
import argparse

from stylegan_xl import dnnlib
import numpy as np
import pandas as pd
import PIL.Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm, trange

from stylegan_xl import legacy
from stylegan_xl.torch_utils import gen_utils
from memnet import MemNet, load_image_mean
from utils import linear_transformation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc_idx', type=int, required=True)
    args = parser.parse_args()

    sc_idx = args.sc_idx

    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
    device = torch.device('cuda')

    # load pretrained StyleGAN-XL
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.eval().requires_grad_(False).to(device)
    print('Loaded networks from "%s"' % network_pkl)

    model = MemNet()
    checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    superclasses = np.load('data/imagenet_superclass/superclasses.npy', allow_pickle=True).item()
    superclass_keys = []
    for k, v in superclasses.items():
        key = k.lower().replace(' ', '_').replace(',', '').replace('.', '')
        superclass_keys.append(key)

    class_name = superclass_keys[sc_idx]

    data_dir = "data/imagenet_superclass/"
    save_dir = data_dir+f"{sc_idx}-{class_name}/"
    os.makedirs(save_dir, exist_ok=True)
    print(f"SUPERCLASS save_dir: {save_dir}")

    if sc_idx in [8, 9, 10]:
        dlats = np.load(f"/h/hwhan14/memorability-stylegan-xl/data/imagenet_superclass/{class_name}_100k/imagenet256_dlats_sgxl.npy")
    else:
        dlats = np.load(save_dir+"imagenet256_dlats_sgxl.npy")
    print(f"dlats.shape: {dlats.shape}")

    img_mean = load_image_mean()
    img_transform = transforms.Compose([
        transforms.Resize((256,256), PIL.Image.BILINEAR),
        lambda x: np.array(x),
        lambda x: np.subtract(x[:,:,[2, 1, 0]], img_mean),
        transforms.ToTensor(),
        lambda x: np.array([np.array(y) for y in transforms.functional.ten_crop(x, (227, 227))]),
    ])

    memorability_scores = []
    pbar = tqdm(dlats)

    for dlat in pbar:
        w = torch.tensor(dlat).to(device)
        
        img = gen_utils.w_to_img(G, w, to_np=True).squeeze()
        img = PIL.Image.fromarray(img, "RGB")
    
        imgs = img_transform(img)
        imgs = torch.Tensor(imgs).to(device)
    
        model.eval()
        with torch.no_grad():
            out = model(imgs)
            out = out.cpu().numpy().squeeze()
    
        score = linear_transformation(out.mean())
        pbar.set_postfix(score=score)
        memorability_scores.append(score)

    np.save(save_dir+"imagenet256_memorability_memnet.npy", memorability_scores)