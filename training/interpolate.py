import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
import pretrained_networks

save_path = "G:/training_datasets/interpolate_ffhq"
def save_images(images, i):
    for img_array in images:
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(save_path, f'frame_%2d.png') % (i), 'PNG')

# Initialize generator and perceptual model
tflib.init_tf()

generator_network, discriminator_network, Gs = pretrained_networks.load_networks("dummy")
dim = 512
STEPS = 3

def override(dim_range, start, end, tar_dlatents, src_dlatents):
    for i in dim_range:
        for j in range(start, end):
            tar_dlatents[:, i, j] = src_dlatents[:, i, j]
    return tar_dlatents

def gen_movie(src, tar, count):
    #count = 121
    #src = "{0:0=2d}".format(src)
    #tar = "{0:0=2d}".format(tar)
    print("src and target = ", src, tar, count)
    original = np.load('/results/test_changed/order/0.npy')
    #person_a = original
    person_a = np.load('/results/test_changed/order/%d.npy' % src)
    person_b = np.load('/results/test_changed/order/%d.npy' % tar)
    z = np.empty((STEPS, 18, dim))
    for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=STEPS)):
        z[i] = alpha * person_b + (1.0-alpha) * person_a
    #for c in np.linspace(0, 1, 30):
    #    generate_image(c*person_a + (1-c)*person_b)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    
    for i in range(STEPS):
        print("l.shape", z[i:i+1].shape)
        npy = z[i:i+1]
        #npy = override(range(6, 18), 0, 512, z[i:i+1], original)
        #images = Gs.components.synthesis.run(original, **synthesis_kwargs)
        #images = Gs.components.synthesis.run(z[i:i+1], minibatch_size=1, randomize_noise=False, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), structure='fixed')
        images = Gs.components.synthesis.run(npy, minibatch_size=1, randomize_noise=False, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), structure='fixed')
        save_images(images, count)
        count += 1

for i in range(11):
    gen_movie(i, i+1, i * STEPS)
#gen_movie(0, 2, STEPS)

def test():
    qlatent1 = np.random.randn(512)[None, :]
    qlatent2 = np.random.randn(512)[None, :]
    
    dlatent1 = np.load('/latent_representations_emotion/s006-01_jpg_01.npy')
    #dlatent1 = Gs.components.mapping.run(qlatent1, None)
    dlatent2 = Gs.components.mapping.run(qlatent2, None)
    
    qlatents = np.vstack([(1 - i) * qlatent1 + i * qlatent2 for i in np.linspace(0, 1, 30)])
    dlatents = np.vstack([(1 - i) * dlatent1 + i * dlatent2 for i in np.linspace(0, 1, 30)])
    
    #dqlatents = Gs.components.mapping.run(qlatents, None)

    dimages = Gs.components.synthesis.run(dlatents, **synthesis_kwargs)
    #dqimages = Gs.components.synthesis.run(dqlatents, **synthesis_kwargs)
    #qimages = Gs.run(qlatents, None)
    save_images(qimages)