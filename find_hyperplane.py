import os

import numpy as np
from tqdm import tqdm, trange

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


if __name__ == "__main__":
    data_dir = "/home/hhan228/memorability/Willow/per_class/"
    save_dir = "data/per_class/"

    all_class_indices = sorted([int(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    exist_class_indices = sorted([int(f.split("-")[0]) for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))])

    class_indices = sorted(list(set(all_class_indices) - set(exist_class_indices)))
    print("class_indices:", class_indices)
    print("len(class_indices):", len(class_indices))

    for cidx in tqdm(class_indices):
        dlats = np.load(data_dir+f"{cidx}/imagenet256_dlats_memnet.npy")
        mem_scores = np.load(data_dir+f"{cidx}/imagenet256_memorability_memnet.npy").reshape(-1, 1)
        
        print("dlats shape:", dlats.shape)
        print("mem_scores shape:", mem_scores.shape)
        
        dlats = dlats.reshape((dlats.shape[0]*dlats.shape[1], dlats.shape[2]*dlats.shape[3]))
        print("dlats shape:", dlats.shape)

        mem_mean = np.mean(mem_scores)
        y = np.ones_like(mem_scores)
        y[mem_scores < mem_mean] = 0
        
        X_train, X_test, y_train, y_test = train_test_split(dlats, np.ravel(y), test_size=0.1, random_state=42)
        
        clf = LogisticRegression(max_iter=5000)
        clf.fit(X_train, y_train)
        
        pred = clf.predict(X_train)
        print("train accuracy:", metrics.accuracy_score(y_train, pred))
        
        pred = clf.predict(X_test)
        print("test accuracy:", metrics.accuracy_score(y_test, pred))
        
        hyperplane = clf.coef_[0]
        np.save(save_dir+f'{cidx}-imagenet256_hyperplane_memnet.npy', hyperplane)
