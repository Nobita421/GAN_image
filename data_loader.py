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

    dataset_source = cfg.get('dataset_source', 'folder')
    if dataset_source == 'tfds_cifar10':
        try:
            import tensorflow_datasets as tfds
        except Exception as e:
            raise RuntimeError(
                "dataset_source=tfds_cifar10 requires tensorflow_datasets. "
                "Install it with: pip install tensorflow-datasets"
            ) from e

        ds = tfds.load('cifar10', split='train', as_supervised=True, shuffle_files=True)

        def map_img(img, _label):
            img = tf.image.resize(img, (size, size), method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32)
            img = (img / 127.5) - 1.0
            if channels == 1:
                img = tf.image.rgb_to_grayscale(img)
            else:
                img = tf.ensure_shape(img, (size, size, 3))
            return img

        ds = ds.map(map_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=10000).batch(bs).prefetch(tf.data.AUTOTUNE)
        return ds

    dataset_path = dataset_path or cfg.get('dataset_path')
    files = []
    for root, _, filenames in os.walk(dataset_path):
        for fn in filenames:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append(os.path.join(root, fn))
    files = sorted(files)

    train_split = float(cfg.get('train_split', 1.0))
    if 0 < train_split < 1.0 and len(files) > 0:
        n_train = max(1, int(len(files) * train_split))
        files = files[:n_train]

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
