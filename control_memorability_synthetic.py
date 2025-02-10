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
from sklearn.model_selection import train_test_split

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


def control_memorability(flatten_w, G, hp, coeff, cls_id, img_num, img_dir, transform=True, orig_mem_score=None):
    x = np.ravel(flatten_w[img_num]) + coeff * hp
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
        img.save(img_dir+f"class{cls_id}_dlatidx{img_num}_memcoef{coeff}_memscore{score:.4f}.png")
        # plt.imshow(img)
        # plt.show()

    return score


if __name__ == "__main__":
    device = torch.device('cuda:1')

    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
    # load pretrained StyleGAN-XL
    with dnnlib.util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)['G_ema']
        generator = generator.eval().requires_grad_(False).to(device)
    print('Loaded networks from "%s"' % network_pkl)

    model = MemNet()
    checkpoint = torch.utils.model_zoo.load_url("https://github.com/andrewrkeyes/Memnet-Pytorch-Model/raw/master/model.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    save_dir = "data/synthetic_memorability_controlled/"
    hp_dir = "data/per_class/"
    # class_indices = sorted([int(f.split("-")[0]) for f in os.listdir(hp_dir) if os.path.isfile(os.path.join(hp_dir, f))])
    class_indices = [339]
    print("len(class_indices):", len(class_indices))

    df_dict = {
        "imagenet_class_idx": [], "dlat_idx": [],
        -100: [], -95: [], -90: [], -85: [], -80: [], -75: [], -70: [], -65: [], -60: [], -55: [], -50: [], -45: [], -40: [], -35: [], -30: [], -25: [], -20: [], -15: [], -10: [], -5: [],
        0: [],
        5: [], 10: [], 15: [], 20: [], 25: [], 30: [], 35: [], 40: [], 45: [], 50: [], 55: [], 60: [], 65: [], 70: [], 75: [], 80: [], 85: [], 90: [], 95: [], 100: []
    }
    coef_list = list(df_dict.keys())[2:]
    coef_list.remove(0)

    pbar = tqdm(class_indices)
    for i in pbar:
        # dlats = np.load(f"/home/hhan228/memorability/Willow/per_class/{i}/imagenet256_dlats_memnet.npy")
        # dlats = dlats.reshape((dlats.shape[0]*dlats.shape[1], dlats.shape[2]*dlats.shape[3]))
        # mem_scores = np.load(f"/home/hhan228/memorability/Willow/per_class/{i}/imagenet256_memorability_memnet.npy").reshape(-1, 1)

        # mem_mean = np.mean(mem_scores)
        # y = np.ones_like(mem_scores)
        # y[mem_scores < mem_mean] = 0
        
        # _, X_test, _, y_test = train_test_split(dlats, np.ravel(y), test_size=0.1, random_state=42)
        # np.save(f"/home/hhan228/memorability/Willow/per_class/{i}/imagenet256_dlats_test_flatten_memnet.npy", X_test)
        X_test = np.load(f"/home/hhan228/memorability/Willow/per_class/{i}/imagenet256_dlats_test_flatten_memnet.npy")

        all_idx = sorted(list(range(len(X_test))))
        exist_idx = sorted([int(f.split('_')[-1]) for f in os.listdir(save_dir+f"{i}/") if os.path.isdir(os.path.join(save_dir+f"{i}/", f))])
        print(len(list(set(all_idx) - set(exist_idx))))

        rand_idx = np.random.choice(list(set(all_idx) - set(exist_idx)), 20, replace=False)
        for j in rand_idx:
            img_save_dir = save_dir+f"{i}/test_{j}/"

            df_dict["imagenet_class_idx"].append(i)
            df_dict["dlat_idx"].append(j)

            hyperplane = np.load(hp_dir+f"{i}-imagenet256_hyperplane_memnet.npy")
            
            mem_score_0 = control_memorability(X_test, generator, hyperplane, 0, i, j, img_save_dir)
            df_dict[0].append(mem_score_0)

            for coef in coef_list:
                mem_score = control_memorability(X_test, generator, hyperplane, coef, i, j, img_save_dir, transform=True, orig_mem_score=mem_score_0)
                df_dict[coef].append(mem_score)
        
    df = pd.DataFrame(df_dict)
    df.to_csv("data/synthetic_memorability_controlled/singleclass_stepsize5_200_newcate50.csv", index=False)
