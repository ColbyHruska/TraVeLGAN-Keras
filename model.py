from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Input, LeakyReLU, add, Dropout, Lambda, \
    BatchNormalization, Activation, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
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
        self.latent_dim = 256
        self.image_shape = (128, 128, 3)
        self.init = RandomNormal(mean=0.0, stddev=0.02)

        self.data_loader = DataLoader(self.image_shape[0], batch_size)
        self.sample_X = self.data_loader.load_batch(aug=False)[0][:3]

        self.generator = self.UNet()
        self.generator.summary()
        plot_model(self.generator, 'generator.png', show_shapes=True)

        self.siamese = self.Siamese()
        self.siamese.summary()
        plot_model(self.siamese, 'siamese.png', show_shapes=True)

        self.discriminator = self.Discriminator()
        self.discriminator.summary()
        plot_model(self.discriminator, 'discriminator.png', show_shapes=True)

        self.build()

    def build(self):
        xi = Input(shape=self.image_shape)
        xj = Input(shape=self.image_shape)

        yi = Input(shape=self.image_shape)
        yj = Input(shape=self.image_shape)

        g_xi = self.generator(xi)
        g_xj = self.generator(xj)

        fi_prob = self.discriminator(g_xi)
        fj_prob = self.discriminator(g_xj)
        ri_prob = self.discriminator(yi)
        rj_prob = self.discriminator(yj)

        lat_xi = self.siamese(xi)
        lat_xj = self.siamese(xj)
        lat_g_xi = self.siamese(g_xi)
        lat_g_xj = self.siamese(g_xj)

        def cosine_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            return -K.mean(x * y, axis=-1, keepdims=True)

        def cos_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return shape1[0], 1

        # epsilon = K.placeholder(shape=(None, 1, 1, 1))
        # ai_img = Input(shape=self.image_shape, tensor=epsilon * yi + (1 - epsilon) * g_xi)
        # ai_out = self.discriminator(ai_img)
        # aj_img = Input(shape=self.image_shape, tensor=epsilon * yj + (1 - epsilon) * g_xj)
        # aj_out = self.discriminator(aj_img)
        # grad_mixed = K.gradients(ai_out, [ai_img])[0]
        # norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
        # grad_penalty = K.mean(K.square(norm_grad_mixed - 1))
        # penalty_i = 10. * grad_penalty
        # grad_mixed = K.gradients(aj_out, [aj_img])[0]
        # norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
        # grad_penalty = K.mean(K.square(norm_grad_mixed - 1))
        # penalty_j = 10. * grad_penalty

        ri = K.binary_crossentropy(K.ones_like(ri_prob), ri_prob)
        rj = K.binary_crossentropy(K.ones_like(rj_prob), rj_prob)
        r_loss = (ri + rj) / 2.
        fi = K.binary_crossentropy(K.zeros_like(fi_prob), fi_prob)
        fj = K.binary_crossentropy(K.zeros_like(fj_prob), fj_prob)
        f_loss = (fi + fj) / 2.
        dis_loss = (r_loss + f_loss) / 2.
        dis_updates = Adam(lr=0.0002, beta_1=0.5).get_updates(dis_loss, self.discriminator.trainable_weights)
        self.dis_train = K.function([xi, xj, yi, yj], [dis_loss], dis_updates)

        vij = lat_xi - lat_xj
        vij_prime = lat_g_xi - lat_g_xj
        travel_loss = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vij, vij_prime])
        siam_loss = K.max([0, 0.1 - K.sqrt(K.sum(K.square(vij)))]) + travel_loss
        siam_updates = Adam(lr=0.0002, beta_1=0.5).get_updates(siam_loss, self.siamese.trainable_weights + self.generator.trainable_weights)
        self.siam_train = K.function([xi, xj, g_xi, g_xj], [siam_loss], siam_updates)

        fi = K.binary_crossentropy(K.ones_like(fi_prob), fi_prob)
        fj = K.binary_crossentropy(K.ones_like(fj_prob), fj_prob)
        f_loss = (fi + fj) / 2.
        adv_loss = f_loss + travel_loss
        adv_updates = Adam(lr=0.0002, beta_1=0.5).get_updates(adv_loss, self.discriminator.trainable_weights + self.siamese.trainable_weights + self.generator.trainable_weights)
        self.adv_train = K.function([xi, xj], [adv_loss], adv_updates)

    def Discriminator(self):
        input_tensor = Input(shape=self.image_shape)

        x = Conv2D(64, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=self.init)(
            input_tensor)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(1, kernel_size=1, strides=2, padding="same", use_bias=False, kernel_initializer=self.init)(x)
        x = Flatten()(x)
        output_tensor = Dense(1, activation='sigmoid', kernel_initializer=self.init)(x)

        return Model(input_tensor, output_tensor)

    def Siamese(self):
        input_tensor = Input(shape=self.image_shape)

        x = Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=self.init)(input_tensor)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=self.init)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(256, kernel_size=3, strides=1, padding="same", kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(512, kernel_size=4, strides=2, padding="same", kernel_initializer=self.init)(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(1, kernel_size=1, strides=2, padding="same", kernel_initializer=self.init)(x)
        x = Flatten()(x)
        output_tensor = Dense(self.latent_dim, activation=None, kernel_initializer=self.init)(x)

        return Model(input_tensor, output_tensor)

    def UNet(self):
        def conv2d(layer_input, filters=16, strides=1, name=None, f_size=4):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name,
                       kernel_initializer=self.init)(layer_input)
            d = BatchNormalization(momentum=0.9, name=name + "_bn")(d)
            d = Activation('relu')(d)
            return d

        def residual(layer_input, filters=16, strides=1, name=None, f_size=3):
            d = conv2d(layer_input, filters=filters, strides=strides, name=name, f_size=f_size)
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', name=name + "_2",
                       kernel_initializer=self.init)(d)
            d = BatchNormalization(momentum=0.9, name=name + "_bn2")(d)
            d = add([d, layer_input])
            return d

        def conv2d_transpose(layer_input, filters=16, strides=1, name=None, f_size=4):
            u = UpSampling2D(size=(4, 4), interpolation='bilinear')(layer_input)
            u = Conv2D(filters, strides=strides, name=name, kernel_size=f_size, padding='same',
                       kernel_initializer=self.init)(u)
            u = BatchNormalization(momentum=0.9, name=name + "_bn")(u)
            u = LeakyReLU(alpha=0.2)(u)
            return u

        c0 = Input(shape=self.image_shape)
        c1 = conv2d(c0, filters=self.gf, strides=1, name="g_e1", f_size=7)
        c2 = conv2d(c1, filters=self.gf * 2, strides=2, name="g_e2", f_size=3)
        c3 = conv2d(c2, filters=self.gf * 4, strides=2, name="g_e3", f_size=3)

        r1 = residual(c3, filters=self.gf * 4, name='g_r1')
        r2 = residual(r1, self.gf * 4, name='g_r2')
        r3 = residual(r2, self.gf * 4, name='g_r3')
        r4 = residual(r3, self.gf * 4, name='g_r4')
        r5 = residual(r4, self.gf * 4, name='g_r5')
        r6 = residual(r5, self.gf * 4, name='g_r6')
        r7 = residual(r6, self.gf * 4, name='g_r7')
        r8 = residual(r7, self.gf * 4, name='g_r8')
        r9 = residual(r8, self.gf * 4, name='g_r9')

        d1 = conv2d_transpose(r9, filters=self.gf * 2, f_size=6, strides=2, name='g_d1_dc')
        d2 = conv2d_transpose(d1, filters=self.gf, f_size=6, strides=2, name='g_d2_dc')

        output_img = Conv2D(self.channels, kernel_size=7, strides=1, padding='same', activation='tanh', name='out',
                            kernel_initializer=self.init)(d2)

        return Model(inputs=c0, outputs=output_img)

    @staticmethod
    def set_trainable(m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val

    def plot_results(self, epoch):
        fake = self.generator.predict(self.sample_X)
        res = (np.hstack([fake[0, :, :, :], fake[1, :, :, :], fake[2, :, :, :]]) + 1.) / 2.
        plt.clf()
        plt.imshow(res)
        plt.axis('off')
        plt.savefig(f'./RESULTS/images/epoch_{epoch}.jpg')
        plt.close()

    def train(self, num_epochs):
        for epoch in range(num_epochs + 1):
            x, y = self.data_loader.load_batch()
            xi = x[:self.batch_size]
            xj = x[self.batch_size:]
            yi = y[:self.batch_size]
            yj = y[self.batch_size:]

            g_xi = self.generator(xi)
            g_xj = self.generator(xj)

            dis_loss = np.mean(self.dis_train([xi, xj, yi, yj]))
            siam_loss = np.mean(self.siam_train([xi, xj, g_xi, g_xj]))
            adv_loss = np.mean(self.adv_train([xi, xj]))

            print(f"Epoch {epoch}/{num_epochs}:\t[G_loss: {adv_loss}]\t[D_loss: {dis_loss}]\t[S_loss: {siam_loss}]")

            if epoch % 10 == 0:
                self.plot_results(epoch)


if __name__ == '__main__':
    gan = TraVeLGAN(batch_size=12)
    gan.train(num_epochs=40_000)
    print()
