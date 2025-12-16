#!/usr/bin/env python
"""Setup and smoke test for GAN Image Generator project"""
import os
import sys

def create_utils_directory():
    """Create utils directory and files"""
    print("Creating utils directory...")
    os.makedirs('utils', exist_ok=True)
    
    # visualizer.py
    print("Creating utils/visualizer.py...")
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
    
    # metrics.py
    print("Creating utils/metrics.py...")
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
    
    # logger.py
    print("Creating utils/logger.py...")
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
    
    # __init__.py
    print("Creating utils/__init__.py...")
    with open('utils/__init__.py', 'w') as f:
        f.write('# Utils package\n')
    
    print("✓ Utils directory created successfully!\n")

def run_smoke_tests():
    """Run smoke tests on all modules"""
    print("="*60)
    print("RUNNING SMOKE TESTS")
    print("="*60)
    
    errors = []
    
    # Test 1: Import generator
    print("\n[1/6] Testing generator.py...")
    try:
        import generator
        g = generator.build_generator()
        print(f"  ✓ Generator created: {g.count_params()} parameters")
    except Exception as e:
        errors.append(f"Generator error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test 2: Import discriminator
    print("\n[2/6] Testing discriminator.py...")
    try:
        import discriminator
        d = discriminator.build_discriminator()
        print(f"  ✓ Discriminator created: {d.count_params()} parameters")
    except Exception as e:
        errors.append(f"Discriminator error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test 3: Import vanilla_gan
    print("\n[3/6] Testing vanilla_gan.py...")
    try:
        import vanilla_gan
        gan = vanilla_gan.VanillaGAN()
        print(f"  ✓ VanillaGAN created successfully")
        print(f"  - Generator: {gan.G.count_params()} params")
        print(f"  - Discriminator: {gan.D.count_params()} params")
    except Exception as e:
        errors.append(f"VanillaGAN error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test 4: Import data_loader
    print("\n[4/6] Testing data_loader.py...")
    try:
        import data_loader
        cfg = data_loader.load_config()
        print(f"  ✓ Config loaded: {cfg.get('image_size')}x{cfg.get('image_size')} images")
    except Exception as e:
        errors.append(f"DataLoader error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test 5: Import utils
    print("\n[5/6] Testing utils modules...")
    try:
        from utils import visualizer, logger
        print("  ✓ Utils visualizer imported")
        print("  ✓ Utils logger imported")
    except Exception as e:
        errors.append(f"Utils error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Test 6: Test inference module
    print("\n[6/6] Testing inference.py...")
    try:
        import inference
        print("  ✓ Inference module imported")
    except Exception as e:
        errors.append(f"Inference error: {e}")
        print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    if errors:
        print(f"❌ {len(errors)} test(s) failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("✅ All smoke tests passed!")
        print("\nNext steps:")
        print("  1. Create data directory: mkdir data\\celeba_preprocessed")
        print("  2. Add training images to data\\celeba_preprocessed")
        print("  3. Run training: python train.py")
        return True

if __name__ == '__main__':
    print("GAN Image Generator - Setup & Smoke Test")
    print("="*60)
    
    # Create utils directory
    create_utils_directory()
    
    # Run smoke tests
    success = run_smoke_tests()
    
    sys.exit(0 if success else 1)
