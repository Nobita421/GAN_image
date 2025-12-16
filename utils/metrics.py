import numpy as np
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
