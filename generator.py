import tensorflow as tf
from tensorflow.keras import layers


def build_generator(latent_dim=100, image_size=64, channels=3):
    """Simple upsampling generator for 64x64 images."""
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential()
    # project and reshape
    n_nodes = 4 * 4 * 512
    model.add(layers.Dense(n_nodes, input_dim=latent_dim, kernel_initializer=init))
    model.add(layers.ReLU())
    model.add(layers.Reshape((4, 4, 512)))

    # upsample to 8x8
    model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', kernel_initializer=init))
    model.add(layers.ReLU())

    # 16x16
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', kernel_initializer=init))
    model.add(layers.ReLU())

    # 32x32
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', kernel_initializer=init))
    model.add(layers.ReLU())

    # 64x64
    model.add(layers.Conv2DTranspose(channels, kernel_size=4, strides=2, padding='same', kernel_initializer=init))
    model.add(layers.Activation('tanh'))

    return model


if __name__ == '__main__':
    g = build_generator()
    g.summary()
