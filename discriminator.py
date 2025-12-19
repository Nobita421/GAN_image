import tensorflow as tf
from tensorflow.keras import layers


def build_discriminator(image_size=64, channels=3):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    inp = layers.Input((image_size, image_size, channels))

    x = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=init)(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x) # Remove sigmoid for numerical stability (use from_logits=True)

    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


if __name__ == '__main__':
    d = build_discriminator()
    d.summary()
