import os

# Create utils directory
os.makedirs('utils', exist_ok=True)

# Create visualizer.py
with open('utils/visualizer.py', 'w') as f:
    f.write('''import numpy as np
from PIL import Image
import math


def save_image_grid(images, path, rows=4):
    # images: (N, H, W, C), values uint8
    n = images.shape[0]
    cols = math.ceil(n / rows)
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    grid = Image.new('RGB', (cols * w, rows * h))
    for idx in range(n):
        img = images[idx]
        if c == 1:
            pil = Image.fromarray(img.squeeze(), mode='L')
            pil = pil.convert('RGB')
        else:
            pil = Image.fromarray(img)
        r = idx // cols
        col = idx % cols
        grid.paste(pil, (col * w, r * h))
    grid.save(path)


if __name__ == '__main__':
    print('visualizer ready')
''')

# Create metrics.py
with open('utils/metrics.py', 'w') as f:
    f.write('''import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import pairwise_distances


def get_inception_features(images, batch_size=32):
    # images in [-1,1], resize to 299x299 expected by InceptionV3
    imgs = (images + 1.0) * 127.5
    imgs = imgs.astype('float32')
    imgs = tf.image.resize(imgs, (299,299))
    imgs = preprocess_input(imgs)
    model = inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
    feats = model.predict(imgs, batch_size=batch_size)
    return feats


def fid_score(real_feats, fake_feats):
    # approximate FID
    mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
    # compute sqrt of product
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma_r.dot(sigma_f))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum((mu_r - mu_f)**2) + np.trace(sigma_r + sigma_f - 2*covmean)
    return float(fid)
''')

# Create logger.py
with open('utils/logger.py', 'w') as f:
    f.write('''import logging
import sys


def setup_logger(name='GAN', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger


if __name__ == '__main__':
    log = setup_logger()
    log.info('Logger ready')
''')

# Create __init__.py
with open('utils/__init__.py', 'w') as f:
    f.write('# Utils package\n')

print("All utils files created successfully!")
