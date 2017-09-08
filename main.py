import os
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.layers import Input
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from src.nn import DCGAN_G, DCGAN_D
from src.classifier import get_mnist_classifier
from src.inception_score import get_inception_score
from src.image_process import merge_imgs

from matplotlib import pyplot as plt
import seaborn as sns

sns.set(context='paper', style='white', palette='Set2', font_scale=1, color_codes=False,
        rc={'font.family': 'sans-serif', 'font.serif': ['Palatino'], 'font.sans-serif': ['DejaVu Sans'],
            'text.usetex': True})

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

netG = DCGAN_G(img_size, nz, nc, ngf)
netD = DCGAN_D(img_size, nc, ndf, wgan=False)

# Define optimization
netD_real_input = Input(shape=(nc, img_size, img_size))
noisev = Input(shape=(nz,))
netD_fake_input = netG(noisev)

ϵ_input = K.placeholder(shape=(None, 1, 1, 1))
netD_mixed_input = Input(shape=(nc, img_size, img_size),
                         tensor=ϵ_input * netD_real_input + (1 - ϵ_input) * netD_fake_input)

loss_real = K.mean(netD(netD_real_input))
loss_fake = K.mean(netD(netD_fake_input))

grad_mixed = K.gradients(netD(netD_mixed_input), [netD_mixed_input])[0]  # only gradient values, no variables
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
grad_penalty = K.mean(K.square(norm_grad_mixed - 1))

d_loss = loss_fake - loss_real + λ * grad_penalty
g_loss = -loss_fake

# Define training function
d_training_updates = Adam(lr=lr_d, beta_1=0.0, beta_2=0.9).get_updates(netD.trainable_weights, [], d_loss)
netD_train = K.function([netD_real_input, noisev, ϵ_input],
                        [loss_real, loss_fake, d_loss],
                        d_training_updates)
g_training_updates = Adam(lr=lr_g, beta_1=0.0, beta_2=0.9).get_updates(netG.trainable_weights, [], g_loss)
netG_train = K.function([noisev], [g_loss], g_training_updates)

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
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            _Diters = 100  # Pretrain D for 100 epochs
        else:
            _Diters = Diters
        j = 0
        while j < _Diters and i < batches:
            j += 1
            i += 1
            real_data = X_train[np.random.choice(range(len(X_train)), size=batch_size)]
            noise = np.random.normal(size=(batch_size, nz))
            ϵ = np.random.rand(batch_size, 1, 1, 1)
            errD_real, errD_fake, errD = netD_train([real_data, noise, ϵ])

        if gen_iterations % 50 == 0:
            print('[%d/%d][%d/%d][%d] '
                  'Loss_G:%f \t Loss_D:%f \t D_real:%f \t D_fake:%f \t'
                  % (epoch, niter, i, batches, gen_iterations,
                     errG, errD, errD_real, errD_fake))
            fake = netG.predict(fixed_noise)
            mean_confidence, class_entropy, score_mean, score_std = get_inception_score(fake, mnist_classifier)
            inceps_score_log.append((score_mean, score_std))
            confidence_log.append(mean_confidence)
            class__log.append(class_entropy)

            plt.imshow(merge_imgs(fake, 10, 10, transfrom=True))
            plt.axis('off')
            plt.savefig(os.path.join('results', 'imgs', 'WGAN_img_iter_%d.png' % i))

        noise = np.random.normal(size=(batch_size, nz))
        errG, = netG_train([noise])
        gen_iterations += 1

        records.append([errG, errD, errD_real, errD_fake])

store_name = os.path.join('results', 'logs', 'ImprovedWGAN')
records = pd.DataFrame(records, columns=['D loss', 'G loss', 'D real', 'D fake'])
classifier_log = pd.DataFrame({'Confidence': confidence_log, 'Entropy': class_entropy})
records.to_csv(store_name+'_records.csv')
classifier_log.to_csv(store_name+'_classifier.csv')
netG.save_weights(store_name+'_netG.h5')
netD.save_weights(store_name+'_netD.h5')