from keras.layers import Activation, Flatten, Dense, Dropout, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential


def get_cifar10_classifier(weights_file):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.trainable = False
    return model


def get_mnist_classifier(weights_file):
    model = Sequential()
    model.add(Conv2D(64, (2, 2), activation='relu', padding='valid',
                     input_shape=(1, 28, 28)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.trainable = False
    return model
