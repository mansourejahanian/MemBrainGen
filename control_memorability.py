import os

import pickle
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
from stylegan_xl.inversion_utils import project, pivotal_tuning


img_mean = load_image_mean()
img_transform = transforms.Compose([
    transforms.Resize((256,256), PIL.Image.BILINEAR),
    lambda x: np.array(x),
    lambda x: np.subtract(x[:,:,[2, 1, 0]], img_mean),
    transforms.ToTensor(),
    lambda x: np.array([np.array(y) for y in transforms.functional.ten_crop(x, (227, 227))]),
])


def pred_memorability(img, transform=True):
    if transform:
        img = img_transform(img)
        img = torch.from_numpy(img).to(device)
    else:
        img = transforms.functional.center_crop(img, (227, 227))
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        out = model(img)
        out = out.cpu().numpy().squeeze()

    if transform:
        score = linear_transformation(out.mean())
    else:
        score = linear_shift(out)

    return score


def control_memorability(G, hp, coeff, nsd_id, sc_idx, img_dir, transform=True, orig_mem_score=None):
    proj_w = np.load(f"/home/hhan228/memorability/Willow/shared1000_inversions/{nid}/projected_w.npy")
    flatten_w = proj_w.reshape((proj_w.shape[0], proj_w.shape[1]*proj_w.shape[2]))

    x = np.ravel(flatten_w) + coeff * hp
    x = torch.from_numpy(x.reshape((1, 32, 512))).to(device)
    
    img = gen_utils.w_to_img(G, x, to_np=True)[0]
    img = PIL.Image.fromarray(img, "RGB")

    score = pred_memorability(img, transform)

    save_img = False
    if coeff == 0:
        save_img = True
    elif orig_mem_score and (coeff < 0):
        if (orig_mem_score - score) >= 0.02:
            save_img = True
    elif orig_mem_score and (coeff > 0):
        if (score - orig_mem_score) >= 0.02:
            save_img = True

    if save_img:
        os.makedirs(img_dir, exist_ok=True)
        img.save(img_dir+f"{nsd_id}_class{sc_idx}_memcoef{coeff}_memscore{score:.4f}.png")
        # plt.imshow(img)
        # plt.show()

    return score


if __name__ == "__main__":
    device = torch.device('cuda')

    model = MemNet()
    checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    save_dir = "data/memorability_controlled/"
    hp_dir = "data/per_class/"
    class_indices = sorted([int(f.split("-")[0]) for f in os.listdir(hp_dir) if os.path.isfile(os.path.join(hp_dir, f))])
    print("len(class_indices):", len(class_indices))

    nsd_superclass = pd.read_csv("data/imagenet_superclass/shared1000_imagenet_superclass.csv")
    nsd_superclass = nsd_superclass[nsd_superclass["imagenet_class_idx"].isin(class_indices)].copy()
    print("nsd_superclass.shape:", nsd_superclass.shape)

    df_dict = {
        "nsd_id": [], "imagenet_class_idx": [],
        -240: [], -220: [], -200: [], -180: [], -160: [], -140: [], -120: [], -100: [], -80: [], -60: [], -40: [], -20: [],
        0: [],
        20: [], 40: [], 60: [], 80: [], 100: [], 120: [], 140: [], 160: [], 180: [], 200: [], 220: [], 240: []
    }
    coef_list = list(df_dict.keys())[2:]
    coef_list.remove(0)

    pbar = tqdm(nsd_superclass.iterrows(), total=nsd_superclass.shape[0])
    for i, row in pbar:
        nid = row["filename"].split("_")[0]
        scidx = row["imagenet_class_idx"]
        img_save_dir = save_dir+f"{nid}/"

        df_dict["nsd_id"].append(nid)
        df_dict["imagenet_class_idx"].append(scidx)

        hyperplane = np.load(hp_dir+f"{scidx}-imagenet256_hyperplane_memnet.npy")
        with open(f"/home/hhan228/memorability/Willow/shared1000_inversions/{nid}/G.pkl", "rb") as f:
            generator = pickle.load(f)["G_ema"]
            generator = generator.eval().requires_grad_(False).to(device)
        
        mem_score_0 = control_memorability(generator, hyperplane, 0, nid, scidx, img_save_dir)
        df_dict[0].append(mem_score_0)

        for coef in coef_list:
            mem_score = control_memorability(generator, hyperplane, coef, nid, scidx, img_save_dir, transform=True, orig_mem_score=mem_score_0)
            df_dict[coef].append(mem_score)
    
    df = pd.DataFrame(df_dict)
    df.to_csv("data/memorability_controlled/shared1000_singleclass_stepsize20_240.csv", index=False)
