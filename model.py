from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, LeakyReLU, add, Dropout, Lambda, \
    BatchNormalization, Activation, Flatten, Dense, UpSampling2D, Concatenate, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import cosine_similarity
from keras.initializers import RandomNormal
from keras.utils import plot_model
import keras.backend as K
from DataLoader import DataLoader
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class TraVeLGAN:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.gf = 64
        self.channels = 3
        self.latent_dim = 1024
        self.image_shape = (128, 128, 3)

        self.data_loader = DataLoader(self.image_shape[0], batch_size)
        xi, yi = self.data_loader.load_batch(aug=False)
        plt.title('Dataset sample')
        plt.imshow((np.vstack([np.hstack([xi[0], xi[1], xi[2]]), np.hstack([yi[0], yi[1], yi[2]])]) + 1.) / 2.)
        plt.show()
        self.sample_X = xi[:3]

        self.generator12 = self.UNet()
        self.generator12.summary()
        plot_model(self.generator12, 'generator.png', show_shapes=True)
        self.generator21 = self.UNet()

        self.siamese = self.Siamese()
        self.siamese.summary()
        plot_model(self.siamese, 'siamese.png', show_shapes=True)

        self.discriminator = self.Discriminator()
        self.discriminator.summary()
        plot_model(self.discriminator, 'discriminator.png', show_shapes=True)
        self.discriminator = None

        self.build()

    def build(self):
        xi = Input(shape=self.image_shape)
        yi = Input(shape=self.image_shape)

        self.discriminator1 = self.Discriminator()
        self.discriminator2 = self.Discriminator()

        g_x = self.generator12(xi)
        g_y = self.generator21(yi)

        fi_prob = self.discriminator1(g_x)
        ri_prob = self.discriminator1(xi)
        rj_prob = self.discriminator2(yi)
        fj_prob = self.discriminator2(g_y)

        fake_i = K.mean(K.binary_crossentropy(K.zeros_like(fi_prob), fi_prob))
        fake_j = K.mean(K.binary_crossentropy(K.zeros_like(fj_prob), fj_prob))
        real_i = K.mean(K.binary_crossentropy(K.ones_like(ri_prob), ri_prob))
        real_j = K.mean(K.binary_crossentropy(K.ones_like(rj_prob), rj_prob))
        self.discriminator_loss = (fake_i + fake_j + real_i + real_j) / 4.
        self.discriminator_updates = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9).get_updates(self.discriminator_loss,
                                                                                         self.discriminator1.trainable_weights +
                                                                                         self.discriminator2.trainable_weights)
        self.D_train = K.function([xi, yi], [self.discriminator_loss], self.discriminator_updates)

        lat_xi = self.siamese(xi)
        lat_xj = self.siamese(g_x)
        lat_g_xi = self.siamese(yi)
        lat_g_xj = self.siamese(g_y)

        orders = [np.array(list(range(i, self.batch_size)) + list(range(i))) for i in
                  range(1, self.batch_size)]
        losses_S1 = []
        losses_S2 = []
        losses_S3 = []

        def siamese_lossfn(logits, labels=None, diff=False, diffmargin=10., samemargin=0.):
            if diff:
                return tf.maximum(0., diffmargin - tf.reduce_sum(logits, axis=-1))
            return tf.reduce_sum(logits, axis=-1)

        for i, order in enumerate(orders):
            other = tf.constant(order)
            dists_withinx1 = lat_xi - tf.gather(lat_xi, other)
            dists_withinx2 = lat_xj - tf.gather(lat_xj, other)
            dists_withinG1 = lat_g_xi - tf.gather(lat_g_xi, other)
            dists_withinG2 = lat_g_xj - tf.gather(lat_g_xj, other)
            losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx1) ** 2, diff=True)))
            losses_S1.append(tf.reduce_mean(siamese_lossfn((dists_withinx2) ** 2, diff=True)))
            losses_S2.append(tf.reduce_mean((dists_withinx1 - dists_withinG1) ** 2))
            losses_S2.append(tf.reduce_mean((dists_withinx2 - dists_withinG2) ** 2))
            losses_S3.append(tf.reduce_mean(tf.reduce_sum(
                -(tf.nn.l2_normalize(dists_withinx1, axis=[-1]) * tf.nn.l2_normalize(dists_withinG1, axis=[-1])),
                axis=-1)))
            losses_S3.append(tf.reduce_mean(tf.reduce_sum(
                -(tf.nn.l2_normalize(dists_withinx2, axis=[-1]) * tf.nn.l2_normalize(dists_withinG2, axis=[-1])),
                axis=-1)))
        loss_S1 = tf.reduce_mean(losses_S1)
        loss_S2 = tf.reduce_mean(losses_S2)
        loss_S3 = tf.reduce_mean(losses_S3)
        self.siamese_loss = loss_S1 + loss_S2 + loss_S3

        # siamese_updates = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9).get_updates(siamese_loss,
        #                                                                       self.siamese.trainable_weights)
        # self.S_train = K.function([xi, yi], [siamese_loss], siamese_updates)

        adversarial_loss_1 = K.mean(K.binary_crossentropy(K.ones_like(fi_prob), fi_prob))
        adversarial_loss_2 = K.mean(K.binary_crossentropy(K.ones_like(fj_prob), fj_prob))
        self.generator_loss = (adversarial_loss_1 + adversarial_loss_2) + 10. * self.siamese_loss
        self.generator_updates = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9).get_updates(self.generator_loss,
                                                                                     self.generator12.trainable_weights +
                                                                                     self.generator21.trainable_weights)
        self.G_train = K.function([xi, yi], [self.generator_loss, self.siamese_loss], self.generator_updates)

    def Discriminator(self):
        input_tensor = Input(shape=self.image_shape)

        x = Conv2D(64, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(input_tensor)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Flatten()(x)
        output_tensor = Dense(1, activation='sigmoid', kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)

        return Model(input_tensor, output_tensor)

    def Siamese(self):
        input_tensor = Input(shape=self.image_shape)

        x = Conv2D(64, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(input_tensor)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same",
                   kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        output_tensor = Dense(self.latent_dim, activation=None,
                              kernel_initializer=tf.truncated_normal_initializer(0, .02))(x)

        return Model(input_tensor, output_tensor)

    def UNet(self, out_ch=3, start_ch=64, depth=3, inc_rate=2., activation=LeakyReLU(0.2),
             dropout=0.2, batchnorm=True, maxpool=False, upconv=False, residual=True):

        def conv_block(m, dim, acti, bn, res, do=0):
            n = Conv2D(dim, 3, padding='same', kernel_initializer=tf.random_normal_initializer(0, .02))(m)
            n = BatchNormalization(momentum=0.9)(n) if bn else n
            n = LeakyReLU(0.2)(n)
            n = Dropout(do)(n) if do else n
            n = Conv2D(dim, 3, padding='same', kernel_initializer=tf.random_normal_initializer(0, .02))(n)
            n = BatchNormalization(momentum=0.9)(n) if bn else n
            n = LeakyReLU(0.2)(n)
            return Concatenate()([m, n]) if res else n

        def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
            if depth > 0:
                n = conv_block(m, dim, acti, bn, res)
                m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
                m = level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
                if up:
                    m = UpSampling2D()(m)
                    m = Conv2D(dim, 2, padding='same', kernel_initializer=tf.random_normal_initializer(0, .02))(m)
                    m = BatchNormalization(momentum=0.9)(m)
                    m = Activation('relu')(m)
                else:
                    m = Conv2DTranspose(dim, 3, strides=2, padding='same',
                                        kernel_initializer=tf.random_normal_initializer(0, .02))(m)
                    m = BatchNormalization(momentum=0.9)(m)
                    m = Activation('relu')(m)
                n = Concatenate()([n, m])
                m = conv_block(n, dim, acti, bn, res)
            else:
                m = conv_block(m, dim, acti, bn, res, do)
            return m

        i = Input(shape=self.image_shape)
        o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
        o = Conv2D(out_ch, 1, activation='tanh')(o)
        return Model(inputs=i, outputs=o)

    def plot_results(self, epoch):
        fake = self.generator12.predict(self.sample_X)
        res = np.hstack([fake[0], fake[1], fake[2]])
        org = np.hstack([self.sample_X[0], self.sample_X[1], self.sample_X[2]])
        out = (np.vstack([org, res]) + 1.) / 2.
        plt.clf()
        plt.imshow(out)
        plt.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def train(self, num_epochs):
        for epoch in range(num_epochs + 1):
            x, y = self.data_loader.load_batch()
            xi = x[:self.batch_size]
            yi = y[:self.batch_size]

            dis_loss = self.D_train([xi, yi])
            gen_loss, siam_loss = self.G_train([xi, yi])

            print(f"Epoch {epoch}/{num_epochs}:\t[G_loss: {gen_loss}]\t[D_loss: {dis_loss}]\t[S_loss: {siam_loss}]")

            if epoch % 25 == 0:
                self.plot_results(epoch)
            if epoch > 100000 and epoch % 250 == 0:
                self.generator12.save_weights(f'./RESULTS/weights/epoch_{epoch}.h5')


if __name__ == '__main__':
    gan = TraVeLGAN(batch_size=8)
    gan.train(num_epochs=40_000)
    print()
