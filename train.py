import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from vanilla_gan import VanillaGAN
from data_loader import prepare_dataset, load_config
from utils.visualizer import save_image_grid


def train():
    cfg = load_config()
    ds = prepare_dataset(cfg['dataset_path'], 'config.yaml')
    gan = VanillaGAN(cfg)

    latent_dim = cfg['latent_dim']
    epochs = cfg['epochs']
    sample_interval = cfg['sample_interval']
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg['samples_dir'], exist_ok=True)

    real_label = np.ones((cfg['batch_size'], 1)) * 0.9  # label smoothing
    fake_label = np.zeros((cfg['batch_size'], 1))

    step = 0
    for epoch in range(1, epochs + 1):
        prog = tqdm(ds, desc=f'Epoch {epoch}/{epochs}', unit='batch')
        for real_imgs in prog:
            bs = real_imgs.shape[0]
            # sample noise
            z = np.random.normal(0, 1, (bs, latent_dim))
            fake_imgs = gan.G.predict(z, verbose=0)

            # train discriminator
            d_loss_real = gan.D.train_on_batch(real_imgs, np.ones((bs,1))*0.9)
            d_loss_fake = gan.D.train_on_batch(fake_imgs, np.zeros((bs,1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator via combined model
            z2 = np.random.normal(0, 1, (bs, latent_dim))
            g_loss = gan.combined.train_on_batch(z2, np.ones((bs,1)))

            if step % 50 == 0:
                prog.set_postfix({'d_loss': float(d_loss[0]), 'g_loss': float(g_loss)})
            step += 1

        # save samples and checkpoint
        if epoch % sample_interval == 0 or epoch == 1:
            n_samples = 16
            z = np.random.normal(size=(n_samples, latent_dim))
            gen = gan.G.predict(z)
            # denormalize
            gen = (gen + 1.0) * 127.5
            save_image_grid(gen.astype('uint8'), f"{cfg['samples_dir']}/epoch_{epoch:03d}.png", rows=4)
            gan.save(save_dir)

    # final save
    gan.save(save_dir)


if __name__ == '__main__':
    train()
