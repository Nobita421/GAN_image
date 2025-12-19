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
        # ... (existing tfds code)
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
    
    # Optimized folder loading using tf.data pipeline
    def load_and_preprocess(path):
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=channels, expand_animations=False)
            img = tf.image.resize(img, (size, size))
            img = (tf.cast(img, tf.float32) / 127.5) - 1.0
            # Ensure shape is set for the batching
            img = tf.ensure_shape(img, (size, size, channels))
            return img
        except Exception:
            # Return a zero tensor if decoding fails (rare with decode_image)
            return tf.zeros((size, size, channels))

    # Filter for common image extensions
    import glob
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = []
    for ext in valid_exts:
        all_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(dataset_path, f"*{ext.upper()}")))
    
    if not all_files:
        raise RuntimeError(f"No images found in {dataset_path}")
        
    ds = tf.data.Dataset.from_tensor_slices(all_files).shuffle(len(all_files))
    
    train_split = float(cfg.get('train_split', 1.0))
    if 0 < train_split < 1.0:
        num_files = tf.data.experimental.cardinality(ds).numpy()
        if num_files > 0:
            ds = ds.take(int(num_files * train_split))

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(bs, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    cfg = load_config()
    ds = prepare_dataset(cfg['dataset_path'])
    for batch in ds.take(1):
        print('Batch shape:', batch.shape)
