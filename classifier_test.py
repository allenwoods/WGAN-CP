from keras.datasets import cifar10, mnist
from keras.utils import to_categorical
(cifar_x_train, cifar_y_train), (cifar_x_test, cifar_y_test) = cifar10.load_data()
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()
print("CIFAR10 Dataset:"%cifar_x_train.shape)
print("MNIST Dataset:"%mnist_x_train.shape)
