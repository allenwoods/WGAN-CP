import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd

from src.gpu_utils import setup_one_gpu
setup_one_gpu()

import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from src.nn import WGAN_G, WGAN_D, DCGAN_G, DCGAN_D
from src.classifier import get_mnist_classifier
from src.inception_score import get_inception_score
from src.image_process import merge_imgs

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(context='paper', style='white', palette='Set2', font_scale=1, color_codes=False,
        rc={'font.family': 'sans-serif', 'font.serif': ['Palatino'], 'font.sans-serif': ['DejaVu Sans'],
            'text.usetex': False})

# Prevent  Resource exhausted:OOM error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
K.set_image_data_format('channels_first')  # (channel, row, col)
K.set_image_dim_ordering('th')

# Load Datasets
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
mnist_data = np.concatenate([np.expand_dims(mnist_x_train, 1),
                             np.expand_dims(mnist_x_test, 1)], axis=0)
mnist_data = mnist_data.clip(0, 255) / 255
mnist_label = to_categorical(np.concatenate([mnist_y_train, mnist_y_test]), num_classes=10)

X_train = mnist_data

# Set up Networks
nc = 1
nz = 100
ngf = 64
ndf = 64
n_extra_layers = 0
λ = 10

img_size = 28
batch_size = 100
lr_d = 1e-4
lr_g = 1e-4

netG = DCGAN_G()
netD = DCGAN_D()

# Define optimization
netD_real_input = Input(shape=(nc, img_size, img_size))
noisev = Input(shape=(nz,))
netD_fake_input = netG(noisev)

loss_real = netD(netD_real_input)
loss_fake = netD(netD_fake_input)

g_loss = -K.mean(K.log(loss_fake+K.epsilon()))
d_loss = -K.mean(K.log(1 - loss_fake) + .1 * K.log(1 - loss_real) + .9 * K.log(loss_real+K.epsilon()))

# Define training function
d_training_updates = SGD(lr=0.0005, momentum=0.9, nesterov=True).get_updates(netD.trainable_weights, [], d_loss)
netD_train = K.function([netD_real_input, noisev, K.learning_phase()],
                        [loss_real, loss_fake, d_loss],
                        d_training_updates)
g_training_updates = SGD(lr=0.0005, momentum=0.9, nesterov=True).get_updates(netG.trainable_weights, [], g_loss)
netG_train = K.function([noisev, K.learning_phase()], [g_loss], g_training_updates)

# Define Logs and Parameters
inceps_score_log = []
confidence_log = []
class__log = []
records = []
niter = 100
Diters = 5
gen_iterations = 0
errG = 0

mnist_classifier_weights = os.path.join('weights', 'mnist_classifier_weights.h5')
mnist_classifier = get_mnist_classifier(mnist_classifier_weights)

# Define Runing Process

fixed_noise = np.random.normal(size=(batch_size, nz)).astype('float32')
for epoch in range(niter):
    i = 0
    #  每個 epoch 洗牌一下
    np.random.shuffle(X_train)
    batches = X_train.shape[0] // batch_size
    while i < batches:
        _Diters = Diters
        j = 0
        while j < _Diters and i < batches:
            j += 1
            i += 1
            real_data = X_train[np.random.choice(range(len(X_train)), size=batch_size)]
            noise = np.random.normal(size=(batch_size, nz))
            ϵ = np.random.rand(batch_size, 1, 1, 1)
            errD_real, errD_fake, errD = netD_train([real_data, noise, True])

        if gen_iterations % 50 == 0:
            fake = netG.predict(fixed_noise)
            mean_confidence, class_entropy, score_mean, score_std = get_inception_score(fake, mnist_classifier)
            inceps_score_log.append((score_mean, score_std))
            confidence_log.append(mean_confidence)
            class__log.append(class_entropy)
            print('[%d/%d][%d/%d][%d]\n'
                  'Loss_G:%f \t Loss_D:%f \t \n'
                  'Mean Confidence:%f \t Entropy:%f.'
                  % (epoch, niter, i, batches, gen_iterations,
                     errG, errD,
                     mean_confidence, class_entropy))

            x_lim = range(len(confidence_log))
            confidence_log_np = np.array(confidence_log)
            class__log_np = np.array(class__log)
            plt.figure()
            plt.plot(x_lim, confidence_log, 'r')
            plt.fill_between(x_lim, confidence_log_np-class__log_np, confidence_log_np+class__log_np)
            plt.savefig(os.path.join('results', 'imgs', 'mean_confidence_entropy.png'))
            plt.close()

        if gen_iterations % 500 == 0:
            plt.figure()
            plt.imshow(merge_imgs(fake, 10, 10, transfrom=False)[0])
            plt.axis('off')
            plt.savefig(os.path.join('results', 'imgs', 'DCGAN_img_iter_%d.png' % i))
            plt.close()

        noise = np.random.normal(size=(batch_size, nz))
        errG, = netG_train([noise, True])
        gen_iterations += 1

        records.append([errG, errD, errD_real, errD_fake])

store_name = os.path.join('results', 'logs', 'DCGAN')
records = pd.DataFrame(records, columns=['D loss', 'G loss', 'D real', 'D fake'])
classifier_log = pd.DataFrame({'Confidence': confidence_log,
                               'Entropy': class_entropy,
                               'Inception Score': inceps_score_log})
records.to_csv(store_name + '_records.csv')
classifier_log.to_csv(store_name + '_classifier.csv')
netG.save_weights(store_name + '_netG.h5')
netD.save_weights(store_name + '_netD.h5')
