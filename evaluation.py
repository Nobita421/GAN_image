import numpy as np
import os
import tensorflow as tf
from data_loader import prepare_dataset, load_config
from vanilla_gan import VanillaGAN
from utils.metrics import get_inception_features, fid_score


def sample_generated(gan, n=100, latent_dim=100):
    z = np.random.normal(size=(n, latent_dim))
    gen = gan.G.predict(z)
    return gen


def evaluate():
    cfg = load_config()
    gan = VanillaGAN(cfg)
    # load weights if exist
    try:
        gan.G = tf.keras.models.load_model(f"{cfg['save_dir']}/G_final.keras")
    except Exception as e:
        print('Could not load G_final.keras', e)

    # collect real features
    ds = prepare_dataset(cfg['dataset_path'], 'config.yaml')
    real_samples = []
    for i, batch in enumerate(ds.take(10)):
        real_samples.append(batch.numpy())
    real_samples = np.concatenate(real_samples, axis=0)
    if real_samples.shape[-1] != 3:
        # replicate channel
        real_samples = np.repeat(real_samples, 3, axis=-1)

    real_feats = get_inception_features(real_samples[:200])

    fake = sample_generated(gan, n=200, latent_dim=cfg['latent_dim'])
    if fake.shape[-1] != 3:
        fake = np.repeat(fake, 3, axis=-1)
    fake_feats = get_inception_features(fake)

    fid = fid_score(real_feats, fake_feats)
    print('FID (proxy):', fid)

if __name__ == '__main__':
    evaluate()
