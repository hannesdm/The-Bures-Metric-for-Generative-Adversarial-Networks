import os
import sys
import tarfile
import urllib
import numpy as np
from PIL import Image

stl10_dir = 'data/stl10/'


train_images = None
train_labels = None
test_images = None
test_labels = None

def load_stl10_images(download_dir = stl10_dir, format='channels_first', flatten=True, normalize=False, cache=True, resize_shape=None):
    assert format in ('channels_first', 'channels_last')
    global train_images,train_labels, test_images, test_labels

    if cache and train_images is not None and train_labels is not None and test_images is not None and test_labels is not None:
        print("-----Using cached STL10-----")
        return train_images,train_labels, test_images, test_labels

    URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    filename = URL.split('/')[-1]
    filepath = os.path.join(download_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))

        filepath, _ = urllib.request.urlretrieve(URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(stl10_dir)
    train_images = np.fromfile(stl10_dir + 'stl10_binary/unlabeled_X.bin', dtype=np.uint8)
    train_images = train_images.reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1)).astype('float32')
    width = train_images.shape[1]
    height = train_images.shape[2]
    channels = train_images.shape[3]
    if resize_shape is not None:
        train_images = resize_images(train_images, new_size=resize_shape)
        width = resize_shape[0]
        height = resize_shape[1]
    if format == 'channels_first':
        train_images = train_images.transpose((0, 3, 1, 2))
    if flatten:
        train_images = train_images.reshape((-1, channels*width*height))
    if normalize:
        train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_labels = np.asarray([0]*train_images.shape[0], dtype=np.uint8) # fake labels

    test_images = np.fromfile(stl10_dir + 'stl10_binary/test_X.bin', dtype=np.uint8)
    test_images = test_images.reshape((-1, channels, 96, 96)).transpose((0, 3, 2, 1)).astype(
        'float32')
    if resize_shape is not None:
        test_images = resize_images(test_images, new_size=resize_shape)
    if format == 'channels_first':
        test_images = test_images.transpose((0, 3, 1, 2))
    if flatten:
        test_images = test_images.reshape((-1, channels*width*height))
    if normalize:
        test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    test_labels = np.fromfile(stl10_dir + 'stl10_binary/test_y.bin', dtype=np.uint8)
    return train_images,train_labels, test_images, test_labels


def resize_images(X, new_size=(48,48), method=Image.LANCZOS):
    assert len(X.shape) == 4
    assert X[0].min() >= 0 and X[0].max() > 5
    arr = np.zeros((X.shape[0],) + new_size + (X.shape[3],))
    for i in range(X.shape[0]):
        im = Image.fromarray(X[i].astype('uint8'))
        im = im.resize((48,48),method) #, Image.LANCZOS
        arr[i] = np.array(im)
    return arr

def stl_generator(batch_size, images, labels):
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, resize_shape=None):
    train_images,train_labels, test_images, test_labels = load_stl10_images(resize_shape=resize_shape)

    return (
        stl_generator(batch_size, train_images, train_labels ),
        stl_generator(batch_size, test_images, test_labels),
    )
