import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

from data_loader import prepare_dataset, load_config
from vanilla_gan import VanillaGAN
from utils.visualizer import save_image_grid
from utils.metrics import fid_score


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_uint8(images: np.ndarray) -> np.ndarray:
    # images expected in [-1, 1]
    arr = (images + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _take_n_images(ds: tf.data.Dataset, n: int) -> np.ndarray:
    batches = []
    total = 0
    for batch in ds:
        b = batch.numpy()
        batches.append(b)
        total += b.shape[0]
        if total >= n:
            break
    if not batches:
        return np.zeros((0, 0, 0, 0), dtype=np.float32)
    imgs = np.concatenate(batches, axis=0)[:n]
    return imgs


def _load_generator(cfg: dict) -> tf.keras.Model:
    gan = VanillaGAN(cfg)
    g_path = f"{cfg.get('save_dir', './checkpoints')}/G_final.keras"
    try:
        gan.G = tf.keras.models.load_model(g_path)
        print(f"Loaded generator: {g_path}")
    except Exception as e:
        print(f"Could not load {g_path}; using randomly initialized generator. ({e})")
    return gan.G


def _sample_fake_images(generator: tf.keras.Model, n: int, latent_dim: int, batch: int = 128) -> np.ndarray:
    out = []
    remaining = n
    while remaining > 0:
        bs = min(batch, remaining)
        z = np.random.normal(size=(bs, latent_dim)).astype(np.float32)
        imgs = generator.predict(z, verbose=0)
        out.append(imgs)
        remaining -= bs
    return np.concatenate(out, axis=0)


def _get_strategy(cfg: dict):
    use_multi_gpu = bool(cfg.get('use_multi_gpu', True))
    gpus = tf.config.list_physical_devices('GPU')
    if use_multi_gpu and len(gpus) >= 2:
        print(f"Multi-GPU enabled: {len(gpus)} GPUs detected")
        return tf.distribute.MirroredStrategy()
    if len(gpus) == 1:
        print("Single GPU detected; using default strategy")
    else:
        print("No GPU detected; using CPU")
    return tf.distribute.get_strategy()


def _build_inception_feature_extractor(strategy):
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    with strategy.scope():
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    return model


def _inception_features(images: np.ndarray, model: tf.keras.Model, batch_size: int) -> np.ndarray:
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    imgs = _to_uint8(images).astype(np.float32)
    imgs = tf.image.resize(imgs, (299, 299))
    imgs = preprocess_input(imgs)
    feats = model.predict(imgs, batch_size=batch_size, verbose=0)
    return feats


def _realism_proxy_nn(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Realism proxy: mean distance from each fake feature to its nearest real feature."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(real_feats)
    dists, _ = nn.kneighbors(fake_feats, return_distance=True)
    return float(np.mean(dists))


def _diversity_proxy(fake_feats: np.ndarray, max_points: int = 512) -> float:
    """Diversity proxy: mean pairwise distance on a subset of fake features."""
    from sklearn.metrics import pairwise_distances

    n = fake_feats.shape[0]
    if n <= 1:
        return 0.0
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        feats = fake_feats[idx]
    else:
        feats = fake_feats
    d = pairwise_distances(feats)
    # mean of upper triangle without diagonal
    triu = d[np.triu_indices_from(d, k=1)]
    return float(np.mean(triu))


def _plot_loss_curves_if_available(cfg: dict, figures_dir: str) -> Optional[str]:
    import csv
    import matplotlib.pyplot as plt

    save_dir = cfg.get('save_dir', './checkpoints')
    log_path = os.path.join(save_dir, 'training_history.csv')
    if not os.path.exists(log_path):
        return None

    epochs = []
    d_loss = []
    g_loss = []
    with open(log_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            d_loss.append(float(row['d_loss']))
            g_loss.append(float(row['g_loss']))

    if not epochs:
        return None

    out_path = os.path.join(figures_dir, 'loss_curves.png')
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, d_loss, label='D loss')
    plt.plot(epochs, g_loss, label='G loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _save_tsne(real_feats: np.ndarray, fake_feats: np.ndarray, figures_dir: str, max_each: int = 500) -> str:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    n_r = min(real_feats.shape[0], max_each)
    n_f = min(fake_feats.shape[0], max_each)
    real_sub = real_feats[:n_r]
    fake_sub = fake_feats[:n_f]
    X = np.concatenate([real_sub, fake_sub], axis=0)
    y = np.array([0] * n_r + [1] * n_f)

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30)
    emb = tsne.fit_transform(X)

    out_path = os.path.join(figures_dir, 'tsne_real_vs_fake.png')
    plt.figure(figsize=(6, 6))
    plt.scatter(emb[y == 0, 0], emb[y == 0, 1], s=8, alpha=0.6, label='real')
    plt.scatter(emb[y == 1, 0], emb[y == 1, 1], s=8, alpha=0.6, label='fake')
    plt.title('t-SNE of Inception Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _save_latent_interpolation(generator: tf.keras.Model, latent_dim: int, figures_dir: str) -> str:
    # 4 pairs, 8 steps each => 32 images
    pairs = 4
    steps = 8
    z0 = np.random.normal(size=(pairs, latent_dim)).astype(np.float32)
    z1 = np.random.normal(size=(pairs, latent_dim)).astype(np.float32)

    all_z = []
    for i in range(pairs):
        for a in np.linspace(0.0, 1.0, steps):
            all_z.append((1 - a) * z0[i] + a * z1[i])
    all_z = np.stack(all_z, axis=0)
    imgs = generator.predict(all_z, verbose=0)

    out_path = os.path.join(figures_dir, 'latent_interpolation.png')
    save_image_grid(_to_uint8(imgs), out_path, rows=pairs)
    return out_path


@dataclass
class EvalSettings:
    mode: str
    num_real: int
    num_fake: int
    grid_n: int
    inception_batch: int


def _get_eval_settings(cfg: dict) -> EvalSettings:
    mode = (cfg.get('eval_mode') or 'quick').strip().lower()
    if mode not in ('quick', 'full'):
        mode = 'quick'

    if mode == 'full':
        num_real = int(cfg.get('eval_num_real_full', 2000))
        num_fake = int(cfg.get('eval_num_fake_full', 2000))
        grid_n = int(cfg.get('grid_n_full', 64))
    else:
        num_real = int(cfg.get('eval_num_real_quick', 200))
        num_fake = int(cfg.get('eval_num_fake_quick', 200))
        grid_n = int(cfg.get('grid_n_quick', 16))

    inception_batch = int(cfg.get('inception_batch', 64))
    return EvalSettings(mode=mode, num_real=num_real, num_fake=num_fake, grid_n=grid_n, inception_batch=inception_batch)


def evaluate() -> None:
    cfg = load_config()
    settings = _get_eval_settings(cfg)
    figures_dir = cfg.get('figures_dir', 'figures')
    _ensure_dir(figures_dir)

    print(f"Eval mode: {settings.mode}")
    print(f"Using {settings.num_real} real / {settings.num_fake} fake for metrics")

    # real samples
    ds = prepare_dataset(cfg.get('dataset_path'), 'config.yaml')
    real_imgs = _take_n_images(ds.unbatch().batch(cfg.get('batch_size', 64)), settings.num_real)
    if real_imgs.shape[0] == 0:
        raise RuntimeError("No real images found. Check dataset_path/dataset_source and ensure data exists.")

    # generator samples
    G = _load_generator(cfg)
    fake_imgs = _sample_fake_images(G, n=settings.num_fake, latent_dim=int(cfg.get('latent_dim', 100)))

    # grids
    real_grid_path = os.path.join(figures_dir, f"real_grid_{settings.mode}.png")
    fake_grid_path = os.path.join(figures_dir, f"fake_grid_{settings.mode}.png")
    save_image_grid(_to_uint8(real_imgs[: settings.grid_n]), real_grid_path, rows=int(np.sqrt(settings.grid_n)) or 4)
    save_image_grid(_to_uint8(fake_imgs[: settings.grid_n]), fake_grid_path, rows=int(np.sqrt(settings.grid_n)) or 4)

    # feature extractor (+ multi-gpu if available)
    strategy = _get_strategy(cfg)
    inception = _build_inception_feature_extractor(strategy)
    real_feats = _inception_features(real_imgs, inception, batch_size=settings.inception_batch)
    fake_feats = _inception_features(fake_imgs, inception, batch_size=settings.inception_batch)

    fid = fid_score(real_feats, fake_feats)
    realism_proxy = _realism_proxy_nn(real_feats, fake_feats)
    diversity_proxy = _diversity_proxy(fake_feats)

    # qualitative extras
    interp_path = _save_latent_interpolation(G, latent_dim=int(cfg.get('latent_dim', 100)), figures_dir=figures_dir)
    tsne_path = _save_tsne(real_feats, fake_feats, figures_dir=figures_dir)
    loss_curve_path = _plot_loss_curves_if_available(cfg, figures_dir)

    metrics = {
        'eval_mode': settings.mode,
        'num_real': settings.num_real,
        'num_fake': settings.num_fake,
        'grid_n': settings.grid_n,
        'inception_batch': settings.inception_batch,
        'fid_proxy': float(fid),
        'realism_proxy_nn_mean_dist': float(realism_proxy),
        'diversity_proxy_mean_pairwise_dist': float(diversity_proxy),
        'artifacts': {
            'real_grid': real_grid_path,
            'fake_grid': fake_grid_path,
            'latent_interpolation': interp_path,
            'tsne': tsne_path,
            'loss_curves': loss_curve_path,
        },
    }

    metrics_path = os.path.join(figures_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    report_path = os.path.join(figures_dir, 'final_evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Final evaluation report\n\n")
        f.write(f"**Eval mode:** {settings.mode}\\n\n")
        f.write(f"**Counts:** real={settings.num_real}, fake={settings.num_fake}\\n\n")
        f.write("## Quantitative metrics\n")
        f.write(f"- FID (proxy): {fid:.4f}\\n")
        f.write(f"- Realism proxy (NN mean dist): {realism_proxy:.4f}\\n")
        f.write(f"- Diversity proxy (mean pairwise dist): {diversity_proxy:.4f}\\n\n")
        f.write("## Qualitative outputs\n")
        f.write(f"- Real grid: {os.path.basename(real_grid_path)}\\n")
        f.write(f"- Fake grid: {os.path.basename(fake_grid_path)}\\n")
        f.write(f"- Latent interpolation: {os.path.basename(interp_path)}\\n")
        f.write(f"- t-SNE: {os.path.basename(tsne_path)}\\n")
        if loss_curve_path:
            f.write(f"- Loss curves: {os.path.basename(loss_curve_path)}\\n")
        else:
            f.write("- Loss curves: (not available; run training that writes training_history.csv)\\n")

    print("Saved evaluation artifacts to:", figures_dir)
    print("Metrics:", metrics_path)
    print("Report:", report_path)


if __name__ == '__main__':
    evaluate()
