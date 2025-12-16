import os
import numpy as np
import tensorflow as tf
import yaml
from vanilla_gan import VanillaGAN
from PIL import Image


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def denormalize(img):
    # from [-1,1] to [0,255]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def generate(n=16, seed=None, out_dir='./samples'):
    cfg = load_config()
    gan = VanillaGAN(cfg)
    try:
        gan.G = tf.keras.models.load_model(f"{cfg['save_dir']}/G_final.keras")
    except Exception as e:
        print('Failed to load model:', e)

    z = np.random.RandomState(seed).normal(size=(n, cfg['latent_dim'])) if seed else np.random.normal(size=(n, cfg['latent_dim']))
    gen = gan.G.predict(z)
    gen = denormalize(gen)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(gen):
        im = Image.fromarray(img)
        p = f"{out_dir}/sample_{i:03d}.png"
        im.save(p)
        paths.append(p)
    return paths


if __name__ == '__main__':
    print(generate(8))
