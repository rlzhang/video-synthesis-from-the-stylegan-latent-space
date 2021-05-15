import os
import pickle
import dnnlib
import gzip
import json
import numpy as np
from tqdm import tqdm_notebook
import warnings
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

import dnnlib.tflib as tflib
from encoder.generator_model import Generator
import pretrained_networks
import glob
import PIL.Image


def get_microsoft_labeled():
    with open('latent_training_data.pkl', 'rb') as f:
        qlatent_data, dlatent_data, labels_data = pickle.load(f)
    
    X_data = dlatent_data.reshape((-1, 18*512))
    y_age_data = np.array([x['faceAttributes']['age'] for x in labels_data])
    y_gender_data = np.array([x['faceAttributes']['gender'] == 'male' for x in labels_data])
    return X_data, y_age_data, y_gender_data

def get_self_labeled(npy):
    X_data = np.load("style2-emotion-all.npy")
    X_data = X_data.reshape((-1, 18*512))
    y_gender_data = np.load(npy) # dummy gender, may other meanings
    return X_data, y_gender_data

tflib.init_tf()

generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks("dummy")
generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

#X_data, y_age_data, y_gender_data = get_microsoft_labeled()


direction_vectors = "D:/Projects/predicted_npys/stylegan2/*.npy"
dataset = glob.glob(direction_vectors)
for npy in dataset:
    print(npy)
    file_name = os.path.basename(npy)
    X_data, y_gender_data = get_self_labeled(npy)
    print("X_data.shape", X_data.shape, "y_gender_data.shape", y_gender_data.shape)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_data, y_gender_data)
    gender_dircetion = clf.coef_.reshape((18, 512))
    direct_path = "D:/Projects/training_datasets/emotions/style2/%s" % file_name
    np.save(direct_path, gender_dircetion)
    print("saved to ", direct_path)








