import numpy as np

import os
import urllib
import gzip
import pickle

file_names_train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
file_names_test = ['test_batch']

train_x = None
train_labels = None
test_x = None
test_labels = None


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict[b'data'], dict[b'labels']


def cifar_generator(batch_size, data_dir, train=True):
    images, labels = load_cifar_images(data_dir=data_dir, train=train)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load_cifar_images(data_dir, train=True, cache=True):
    # global filenames_train
    # global file_names_test
    global train_x, train_labels, test_x, test_labels
    if cache:
        if train and train_x is not None and train_labels is not None:
            print('--- Using cached train x-------')
            return train_x, train_labels
        elif not train and test_x is not None and test_labels is not None:
            print('--- Using cached test x-------')
            return test_x, test_labels

    if train:
        filenames = file_names_train
    else:
        filenames = file_names_test

    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if train:
        train_x = images
        train_labels = labels
    else:
        test_x = images
        test_labels = labels

    return images, labels


def load(batch_size, data_dir):
    return (
        cifar_generator(batch_size, data_dir, train=True),
        cifar_generator(batch_size, data_dir, train=False)
    )
