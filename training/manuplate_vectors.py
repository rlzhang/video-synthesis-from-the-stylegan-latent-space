import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import time
import cv2
import copy
import glob, re
from encoder.generator_model import Generator
import pretrained_networks

result_dir = "results"
vectors = "./ffhq_dataset/latent_representations/donald_trump_01.npy"
src_vector = np.load(vectors)
dst_seeds = [1370]
cof = 5

def gen_img_with_18_512(Gs, fmt, rnd, dst_seeds=dst_seeds):
    #latents = rnd.randn(1, Gs.input_shape[1])
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    dlatents = Gs.components.mapping.run(latents, None)
    print("1)------------------dlatents.shape", dlatents.shape)
    
    #print("2)------------------dst_latents.shape", dst_latents.shape)
    # mask
    #dlatents = same_image_mask_18_512(dlatents)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
    
    #images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    png_filename = os.path.join(result_dir, 'report/0-original.png')
    cv2.imwrite(png_filename, cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
    return dlatents

def load_vectors(Gs):
    vectors = "D:/Projects/hp/trained_vectors/pre-processed/edit.npy"
    return np.load(vectors)

def gen_img_with_1_512(Gs, fmt, rnd):
    latents = rnd.randn(1, Gs.input_shape[1])
    print("------------------latents.shape", latents.shape)
    images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
    return images

def move_and_show(latent_vector, direction, coeffs):
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
    return new_latent_vector

def same_image_mask_18_512(using_vector):
    assert(using_vector.shape == (1, 18, 512))
    assert(src_vector.shape == (18, 512))
    for i in range(3):
        using_vector[:, i] = src_vector[i]
        #for j in range(300, 512):
        #    using_vector[:, i, j] = src_vector[i, j]
    return using_vector

def replaceRandom(arr, num):
    temp = np.asarray(arr)   # Cast to numpy array
    shape = temp.shape       # Store original shape
    temp = temp.flatten()    # Flatten to 1D
    inds = np.random.choice(temp.size, size=num)   # Get random indices
    temp[inds] = np.random.normal(size=num)        # Fill with something
    temp = temp.reshape(shape)                     # Restore original shape
    return temp

def change_it(vec):
    for i in range(4):
        #vec[:, i] += 0.12
        #arr = vec[:, i, 0:300]
        #vec[:, i, 0:300] = replaceRandom(arr, 300)
        for j in range(0, 300):
           vec[:, i, j] += 0.22
    return vec

def mask(vectors, src):
    for i in range(8):
        #vectors[:, i] = src_dlatents[frame_idx, i]
        for j in range(300, 512):
            vectors[:, i, j] = src[:, i, j]
    return vectors

def override(dim_range, start, end, tar_dlatents, src_dlatents):
    for i in dim_range:
        for j in range(start, end):
            tar_dlatents[:, i, j] = src_dlatents[:, i, j]
    return tar_dlatents

def create_order_npy(npy_name, vectors):
    npy_name = npy_name.split('-')[0]
    np.save(os.path.join(result_dir, 'report/order/%s.npy' % npy_name), vectors)
    print("Saved ", npy_name)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

count = 0
def create_all(vectors, npy, npy_name, Gs, generator, cof):
    global count
    temp = copy.deepcopy(vectors)
    print("====", npy, "=npy_name", npy_name)
    #change cof
    isRight = npy_name.find("-emotion-all-surprise") > 0
    isLeft = npy_name.find("-emotion-turn-left") > 0
    #if isRight:
    #    cof = 5
    if isLeft:
        #cof = -3
        print("====", npy, "=cof", cof)
    vectors = move_and_show(vectors, np.load(npy), [cof])
    np.save(os.path.join(result_dir, 'report/%s.npy' % npy_name), vectors)
    create_order_npy(npy_name, vectors)
    # Restore some of original picture features
    #vectors = mask(vectors, temp)
    vectors = override(range(8, 18), 0, 512, vectors, temp)
    len = vectors.shape[0]
    for i in range(len):
        cur_vector = vectors[i:i+1]
        #images = Gs.components.synthesis.run(cur_vector, randomize_noise=False, **synthesis_kwargs)
        generator.set_dlatents(cur_vector)
        images = generator.generate_images()
        #os.makedirs(result_dir, exist_ok=True)
        png_filename = os.path.join(result_dir, 'report/%s.png' % npy_name)
        #png_filename = os.path.join(result_dir, 'report/interpolate/%d_%s.png' % (count, npy_name))
        count += 1
        #PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
        cv2.imwrite(png_filename, cv2.cvtColor(images[0], cv2.COLOR_RGB2BGR))
    return vectors

def create_full(vectors, npy, file_name_no_extension, Gs, generator):
    STEPS = 20
    for i, alpha in enumerate(np.linspace(start=0.0, stop=cof, num=STEPS)):
        print("---", i, alpha)
        new_vec = create_all(vectors, npy, file_name_no_extension, Gs, generator, alpha)
    return new_vec

def main():
    tflib.init_tf()
    #_G, _D, Gs = pickle.load(open("karras2019stylegan-ffhq-1024x1024.pkl", "rb"))
    _G, _D, Gs = pretrained_networks.load_networks("dummy")
    generator = Generator(Gs, batch_size=1, randomize_noise=False)
    Gs.print_layers()
    rnd = np.random.RandomState(None)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    vectors = gen_img_with_18_512(Gs, fmt, rnd)
    np.save(os.path.join(result_dir, 'report/0-original.npy'), vectors)
    create_order_npy("0-original.npy", vectors)
    # load all directions
    direction_vectors = "D:/Projects/training_datasets/emotions/style2/mixed/*.npy"
    dataset = glob.glob(direction_vectors)
    dataset = natural_sort(dataset)
    for npy in dataset:
        print(npy)
        file_name = os.path.basename(npy)
        file_name_no_extension = os.path.splitext(file_name)[0]
        print(file_name_no_extension)
        #vectors = create_full(vectors, npy, file_name_no_extension, Gs, generator)
        create_all(vectors, npy, file_name_no_extension, Gs, generator, cof)

def main2(seed):
    tflib.init_tf()
    #_G, _D, Gs = pickle.load(open("karras2019stylegan-ffhq-1024x1024.pkl", "rb"))
    _G, _D, Gs = pretrained_networks.load_networks("dummy")
    generator = Generator(Gs, batch_size=1, randomize_noise=False)
    Gs.print_layers()
    rnd = np.random.RandomState(None)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    vectors = gen_img_with_18_512(Gs, fmt, rnd, dst_seeds=seed)
    np.save(os.path.join(result_dir, 'test_changed/0-original.npy'), vectors)
    create_order_npy("0-original.npy", vectors)
    # load all directions
    direction_vectors = "D:/Projects/training_datasets/emotions/style2/*.npy"
    dataset = glob.glob(direction_vectors)
    dataset = natural_sort(dataset)
    for npy in dataset:
        print(npy)
        file_name = os.path.basename(npy)
        file_name_no_extension = os.path.splitext(file_name)[0]
        print(file_name_no_extension)
        #vectors = create_full(vectors, npy, file_name_no_extension, Gs, generator)
        create_all(vectors, npy, file_name_no_extension, Gs, generator, cof)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))