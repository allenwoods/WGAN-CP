from keras.models import Model, Sequential
from keras.layers import Conv2D, BatchNormalization, Input, UpSampling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Flatten, Dense, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)


def WGAN_G(isize, nz, nc, ngf):
    cngf = ngf // 2
    tisize = isize
    while tisize > 7:
        cngf = cngf * 2
        assert tisize % 2 == 0
        tisize = tisize // 2
    _ = inputs = Input(shape=(nz,))
    _ = Reshape((nz, 1, 1))(_)
    _ = Conv2DTranspose(filters=cngf, kernel_size=tisize, strides=1, use_bias=False,
                        kernel_initializer=conv_init,
                        name='initial.{0}-{1}.convt'.format(nz, cngf))(_)
    _ = BatchNormalization(gamma_initializer=gamma_init, momentum=0.9, axis=1, epsilon=1.01e-5,
                           name='initial.{0}.batchnorm'.format(cngf))(_, training=1)
    _ = LeakyReLU(0.2, name='initial.{0}.leaky_relu'.format(cngf))(_)
    csize, cndf = tisize, cngf

    while csize < isize // 2:
        in_feat = cngf
        out_feat = cngf // 2
        _ = Conv2DTranspose(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                            kernel_initializer=conv_init, padding="same",
                            name='pyramid.{0}-{1}.convt'.format(in_feat, out_feat)
                            )(_)
        _ = BatchNormalization(gamma_initializer=gamma_init,
                               momentum=0.9, axis=1, epsilon=1.01e-5,
                               name='pyramid.{0}.batchnorm'.format(out_feat))(_, training=1)

        _ = LeakyReLU(0.2, name='pyramid.{0}.relu'.format(out_feat))(_)
        csize, cngf = csize * 2, cngf // 2
    _ = Conv2DTranspose(filters=nc, kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer=conv_init, padding="same",
                        name='final.{0}-{1}.convt'.format(cngf, nc)
                        )(_)
    outputs = Activation("sigmoid", name='final.{0}.sigmoid'.format(nc))(_)
    return Model(inputs=inputs, outputs=outputs)


def WGAN_D(isize, nc, ndf):
    assert isize % 2 == 0
    _ = inputs = Input(shape=(nc, isize, isize))
    _ = Conv2D(filters=ndf, kernel_size=4, strides=2, use_bias=False,
               padding="same",
               kernel_initializer=conv_init,
               name='initial.conv.{0}-{1}'.format(nc, ndf)
               )(_)
    _ = LeakyReLU(alpha=0.2, name='initial.leakyrelu.{0}'.format(ndf))(_)
    csize, cndf = isize // 2, ndf
    while csize > 7:
        assert csize % 2 == 0
        in_feat = cndf
        out_feat = cndf * 2
        _ = Conv2D(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                   padding="same",
                   kernel_initializer=conv_init,
                   name='pyramid.{0}-{1}.conv'.format(in_feat, out_feat)
                   )(_)
        _ = LeakyReLU(alpha=0.2, name='pyramid.{0}.leakyrelu'.format(out_feat))(_)
        csize, cndf = (csize + 1) // 2, cndf * 2
    _ = Conv2D(filters=1, kernel_size=csize, strides=1, use_bias=False,
               kernel_initializer=conv_init,
               name='final.{0}-{1}.conv'.format(cndf, 1)
               )(_)
    outputs = Flatten()(_)
    return Model(inputs=inputs, outputs=outputs)

def DCGAN_G():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

def DCGAN_D():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(1, 28, 28))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model