import tensorflow as tf
import yaml
from generator import build_generator
from discriminator import build_discriminator


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class VanillaGAN:
    def __init__(self, cfg=None, strategy=None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.latent_dim = cfg['latent_dim']
        self.image_size = cfg['image_size']
        self.channels = cfg['channels']
        self.strategy = strategy or tf.distribute.get_strategy()
        self._build()

    def _build(self):
        with self.strategy.scope():
            self.G = build_generator(self.latent_dim, self.image_size, self.channels)
            self.D = build_discriminator(self.image_size, self.channels)

            # compile discriminator
            self.D.compile(optimizer=tf.keras.optimizers.Adam(
                self.cfg['learning_rate'], 
                beta_1=self.cfg['beta1'],
                clipnorm=1.0  # Gradient clipping for stability
            ),
                           loss='binary_crossentropy', metrics=['accuracy'])

            # combined model
            self.D.trainable = False
            z = tf.keras.Input(shape=(self.latent_dim,))
            img = self.G(z)
            valid = self.D(img)
            self.combined = tf.keras.Model(z, valid)
            self.combined.compile(optimizer=tf.keras.optimizers.Adam(
                self.cfg['learning_rate'], 
                beta_1=self.cfg['beta1'],
                clipnorm=1.0  # Gradient clipping for stability
            ),

    def save(self, save_dir='./checkpoints'):
        self.G.save(f'{save_dir}/G_final.keras')
        self.D.save(f'{save_dir}/D_final.keras')


if __name__ == '__main__':
    gan = VanillaGAN()
    gan.G.summary()
    gan.D.summary()
