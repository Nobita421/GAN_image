import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from vanilla_gan import VanillaGAN
from data_loader import prepare_dataset, load_config
from utils.visualizer import save_image_grid


def _append_training_history(csv_path, epoch, d_loss, g_loss):
    import csv
    header = ['epoch', 'd_loss', 'g_loss']
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([int(epoch), float(d_loss), float(g_loss)])


def get_strategy(cfg):
    use_multi_gpu = bool(cfg.get('use_multi_gpu', True))
    gpus = tf.config.list_physical_devices('GPU')
    if use_multi_gpu and len(gpus) >= 2:
        print(f"Multi-GPU training enabled: {len(gpus)} GPUs detected")
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


def train():
    cfg = load_config()
    strategy = get_strategy(cfg)
    
    ds = prepare_dataset(cfg['dataset_path'], 'config.yaml')
    
    # If using multi-GPU, distribute the dataset
    # if isinstance(strategy, tf.distribute.MirroredStrategy):
    #     ds = strategy.experimental_distribute_dataset(ds)

    gan = VanillaGAN(cfg, strategy=strategy)

    latent_dim = cfg['latent_dim']
    epochs = cfg['epochs']
    sample_interval = cfg['sample_interval']
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg['samples_dir'], exist_ok=True)

    # TensorBoard setup
    log_dir = os.path.join(save_dir, 'logs')
    summary_writer = tf.summary.create_file_writer(log_dir)

    real_label = np.ones((cfg['batch_size'], 1)) * 0.9  # label smoothing
    fake_label = np.zeros((cfg['batch_size'], 1))

    step = 0
    for epoch in range(1, epochs + 1):
        d_losses = []
        g_losses = []
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

            # track losses (for evaluation plots)
            try:
                d_losses.append(float(d_loss[0]))
            except Exception:
                d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            if step % 50 == 0:
                prog.set_postfix({'d_loss': float(d_loss[0]), 'g_loss': float(g_loss)})
                
                # TensorBoard logging
                with summary_writer.as_default():
                    tf.summary.scalar('d_loss', float(d_loss[0]), step=step)
                    tf.summary.scalar('g_loss', float(g_loss), step=step)
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

            # write per-epoch average losses
            if d_losses and g_losses:
                d_mean = float(np.mean(d_losses))
                g_mean = float(np.mean(g_losses))
                _append_training_history(os.path.join(save_dir, 'training_history.csv'), epoch, d_mean, g_mean)

    # final save
    gan.save(save_dir)


if __name__ == '__main__':
    train()
