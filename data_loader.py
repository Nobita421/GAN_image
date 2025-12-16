import os
import yaml
import numpy as np
from PIL import Image
import tensorflow as tf

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_image(img_path, image_size=64, channels=3):
    img = Image.open(img_path)
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img).astype('float32')
    # scale to [-1, 1]
    arr = (arr / 127.5) - 1.0
    if channels == 1:
        arr = np.expand_dims(arr, -1)
    return arr


def prepare_dataset(dataset_path, config_path='config.yaml'):
    cfg = load_config(config_path)
    size = cfg['image_size']
    channels = cfg['channels']
    bs = cfg['batch_size']

    files = []
    for root, _, filenames in os.walk(dataset_path):
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, fn))
    files = sorted(files)

    def gen():
        for p in files:
            try:
                yield preprocess_image(p, image_size=size, channels=channels)
            except Exception as e:
                print('skip', p, e)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(size, size, channels), dtype=tf.float32)
    )
    ds = ds.shuffle(buffer_size=1000).batch(bs).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    cfg = load_config()
    ds = prepare_dataset(cfg['dataset_path'])
    for batch in ds.take(1):
        print('Batch shape:', batch.shape)
