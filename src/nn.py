from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Input
from keras.layers import Conv2DTranspose, Reshape, Activation, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)


def DCGAN_G(isize, nz, nc, ngf):
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


def DCGAN_D(isize, nc, ndf, wgan=True):
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
        if not wgan:  # toggle batchnormalization
            _ = BatchNormalization(name='pyramid.{0}.batchnorm'.format(out_feat),
                                   momentum=0.9, axis=1, epsilon=1.01e-5,
                                   gamma_initializer=gamma_init,
                                   )(_, training=1)
        _ = LeakyReLU(alpha=0.2, name='pyramid.{0}.leakyrelu'.format(out_feat))(_)
        csize, cndf = (csize + 1) // 2, cndf * 2
    if wgan:
        _ = Conv2D(filters=1, kernel_size=csize, strides=1, use_bias=False,
                   kernel_initializer=conv_init,
                   name='final.{0}-{1}.conv'.format(cndf, 1)
                   )(_)
    outputs = Flatten()(_)
    if not wgan:
        outputs = Dense(128)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = LeakyReLU(0.2)(outputs)
        outputs = Dense(2, activation='softmax')(outputs)
    return Model(inputs=inputs, outputs=outputs)
