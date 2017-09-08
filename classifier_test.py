import os
import numpy as np
from keras.datasets import cifar10, mnist
from keras.utils import to_categorical

from src.classifier import get_mnist_classifier, get_cifar10_classifier

(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = cifar10.load_data()
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
cifar_data = np.concatenate([cifar_x_train, cifar_x_test], axis=0)
mnist_data = np.concatenate([np.expand_dims(mnist_x_train, 1),
                             np.expand_dims(mnist_x_test, 1)], axis=0)
cifar_data = cifar_data.clip(0, 255)/255
mnist_data = mnist_data.clip(0, 255)/255

cifar_label = to_categorical(np.concatenate([cifar_y_train, cifar_y_test]), num_classes=10)
mnist_label = to_categorical(np.concatenate([mnist_y_train, mnist_y_test]), num_classes=10)

print("CIFAR10 Dataset:", cifar_data.shape)
print("CIFAR10 Label:", cifar_label.shape)

print("MNIST Dataset:", mnist_data.shape)
print("MNIST Label: ", mnist_label.shape)

cifar_classifier_weights = os.path.join('weights', 'cifar10_classifier_weights.h5')
mnist_classifier_weights = os.path.join('weights', 'mnist_classifier_weights.h5')

cifar10_classifier = get_cifar10_classifier(cifar_classifier_weights)
mnist_classifier = get_mnist_classifier(mnist_classifier_weights)

cifar_eval = cifar10_classifier.evaluate(cifar_data, cifar_label,
                            batch_size=cifar_data.shape[0]//100)
print("CIFAR10 Evaluation:", cifar_eval)
mnist_eval = mnist_classifier.evaluate(mnist_data, mnist_label,
                          batch_size=mnist_data.shape[0]//100)
print("MNIST Evaluation:", mnist_eval)


