import itertools
import os
import sys
import tarfile
import urllib
from abc import ABC, abstractmethod
import numpy as np
import tensorflow_probability as tfp
from numpy.random import default_rng

from .linalg import *

initial_random_seed = 42
stacked_mnist_loc = 'data/stacked_train.npy'
stacked_mnist_y_loc = 'data/stacked_train_labels.npy'
stl10_dir = 'data/stl10/'


def eye(batch_size, dtype='float64'):
    if dtype == 'float64':
        arr = tf.eye(batch_size, dtype='float64')
    else:
        arr = tf.eye(batch_size)
    return arr


def sample_normal(batch, dim, dtype='float64'):
    if dtype == 'float64':
        noise = tf.random.normal([batch, dim], dtype='float64')
    else:
        noise = tf.random.normal([batch, dim])
    return noise


def sample_ring(batch_size, unbalanced=False, n_mixture=8, std=0.01, radius=2.5, random_seed=initial_random_seed,
                mean=(0, 0)):
    """Generate 2D Ring"""
    # thetas = np.linspace(0, 2 * np.pi, n_mixture)
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]  # extra mode compared to gdpp paper
    xs, ys = mean[0] + radius * np.sin(thetas, dtype='float32'), mean[1] + radius * np.cos(thetas, dtype='float32')
    if unbalanced:
        probs = np.logspace(-2, 1, num=n_mixture)
        cat = tfp.distributions.Categorical(probs=probs)
    else:
        cat = tfp.distributions.Categorical(tf.zeros(n_mixture))
    comps = [tfp.distributions.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = tfp.distributions.Mixture(cat, comps)
    return np.asarray(data.sample(batch_size, seed=random_seed))


def sample_grid(batch_size, unbalanced=False, num_components=25, std=0.05, random_seed=initial_random_seed):
    """Generate 2D Grid"""
    if unbalanced:
        probs = np.logspace(-2, 1, num=num_components)
        cat = tfp.distributions.Categorical(probs=probs)
    else:
        cat = tfp.distributions.Categorical(tf.zeros(num_components, dtype=tf.float32))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)
    sigmas = [np.array([std, std]).astype(np.float32) for i in range(num_components)]
    components = list((tfp.distributions.MultivariateNormalDiag(mu, sigma)
                       for (mu, sigma) in zip(mus, sigmas)))
    data = tfp.distributions.Mixture(cat, components)
    return np.asarray(data.sample(batch_size, seed=random_seed))


def sample_batch_uniformTF(batch_size, X):
    """Sample Data Uniform"""
    sizeX = tf.shape(X)
    idx = tf.random.uniform(shape=[batch_size], minval=0, maxval=sizeX[0], dtype=tf.int32)
    return tf.gather(X, idx)

def sample_batch_uniform(batch_size, X, np_generator = None):
    """Sample Data Uniform"""
    if np_generator is None:
        np_generator = default_rng()
    idx = np_generator.choice(X.shape[0],batch_size,replace=False)
    return X[idx]


def load_mnist_images():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    X = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return X


def load_cifar10_images():
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
    X = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return X, train_labels


# labels fine or coarse
def load_cifar100_images(labels='fine'):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar100.load_data(labels)
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
    X = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return X, train_labels


def load_stacked_mnist(stacked_mnist_loc=stacked_mnist_loc, stacked_mnist_labels_loc=stacked_mnist_y_loc):
    images = np.load(stacked_mnist_loc, allow_pickle=True)
    images = images.astype('float32')
    y = np.load(stacked_mnist_labels_loc, allow_pickle=True)
    images = np.rollaxis(images.reshape((-1, 3, 28, 28)), 1, 4)
    return images, y


def load_stl10_images(download_dir = stl10_dir):
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
    train_images = train_images.reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1)).astype('float32') # takes around 10gb of memory
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images


def batch_generator(X, batch_size=32, drop_remainder=False):
    assert batch_size > 0
    i = 0
    while i + batch_size <= X.shape[0]:
        yield X[i:i + batch_size]
        i = i + batch_size
    if i < X.shape[0] and not drop_remainder:  # yield remaining elements smaller than batch_size
        yield X[i:]


def epoch_batch_generator(batch_size, X):
    batch_cache_ds = tf.data.Dataset.from_tensor_slices(X).repeat(None).batch(batch_size=batch_size,
                                                                           drop_remainder=False)
    for batch in batch_cache_ds:
        yield batch


def uniform_sampler_generator(batch_size, X, cache_nr, np_generator = None):
    if batch_size > X.shape[0]:
        batch_size = X.shape[0]
    if cache_nr < batch_size:
        cache_nr = batch_size
    sample_cache = sample_batch_uniform(cache_nr, X, np_generator=np_generator)
    for batch in batch_generator(sample_cache, batch_size, drop_remainder=True):
        yield batch


"sample_cache_nr: integer,'all' or 'batch'. Sets how many samples should be precalculated. " \
"   An integer means how many samples i.e. indices in the first dimension."
"   'all' is a shortcut for X.shape[0] and batch is a shortcut for a cache equal to a single batch size."
"force_equal_batch_size: if true then all batches will have the same number of samples," \
" if false then the remainder will also be a batch i.e. if the number of samples is not divisible by the batch size. "


class Sampler(ABC):
    def __init__(self, batch_size, X, gan, sample_cache_nr='all', force_equal_batch_size=True):
        if batch_size > X.shape[0]:
            batch_size = X.shape[0]
        self.batch_size = batch_size
        self.X = X
        self.gan = gan
        if sample_cache_nr == 'all' or sample_cache_nr is None:
            sample_cache_nr = X.shape[0]
        if sample_cache_nr == 'batch' or sample_cache_nr < batch_size:
            sample_cache_nr = batch_size
        self.sample_cache_nr = sample_cache_nr
        self.force_equal_batch_size = force_equal_batch_size

    @abstractmethod
    def sample_batch(self):
        ...

    def sample(self, nr):
        batches = []
        for _ in range(nr):
            batches.append(self.sample_batch())
        return batches

    @classmethod
    def create(cls, name, batch_size, X, gan, kwargs):
        name = name.lower()
        if name == 'uniform':
            if kwargs is not None:
                return UniformSampler(batch_size, X, gan, **kwargs)
            else:
                return UniformSampler(batch_size, X, gan)
        elif name == 'shuffle':
            if kwargs is not None:
                return ShuffleSampler(batch_size, X, gan, **kwargs)
            else:
                return ShuffleSampler(batch_size, X, gan)
        elif name == 'epoch':
            if kwargs is not None:
                return EpochBatchSampler(batch_size, X, gan, **kwargs)
            else:
                return EpochBatchSampler(batch_size, X, gan)
        elif name == 'npuniform':
            if kwargs is not None:
                return NumpyUniformSampler(batch_size, X, gan, **kwargs)
            else:
                return NumpyUniformSampler(batch_size, X, gan)
        else:
            raise AssertionError('Invalid sampler name.')


class GeneratorSampler(Sampler):
    def __init__(self, batch_size, X, gan, sample_cache_nr='all'):
        super().__init__(batch_size, X, gan, sample_cache_nr)
        self.generator_callable, self.generator_callable_args = self._create_generator()
        self.dataset, self.dataset_iterator = self._create_dataset()

    @abstractmethod
    def _create_generator(self):
        ...

    def _create_dataset(self):
        batch_generator = self.generator_callable
        args = self.generator_callable_args
        dataset = tf.data.Dataset.from_generator(batch_generator, args=args,
                                                 output_shapes=(self.batch_size,) + self.X.shape[1:],
                                                 output_types='float32').repeat(None)
        dataset_iterator = iter(dataset)
        return dataset, dataset_iterator

    def sample_batch(self):
        return next(self.dataset_iterator)


class UniformSampler(GeneratorSampler):

    def __init__(self, batch_size, X, gan, sample_cache_nr='all'):
        super().__init__(batch_size, X, gan, sample_cache_nr)

    def _create_generator(self):
        return uniform_sampler_generator, [self.batch_size, self.X, self.sample_cache_nr]


# infinitly goes through the data sequentially for a specific batch size
class EpochBatchSampler(Sampler):

    def __init__(self, batch_size, X, gan = None, sample_cache_nr='all'):
        super().__init__(batch_size, X, gan, sample_cache_nr)
        self.ds = tf.data.Dataset.from_tensor_slices(X).batch(batch_size=batch_size, drop_remainder=False).repeat(None)
        self.ds_iterator = iter(self.ds)

    def sample_batch(self):
        return next(self.ds_iterator)


class DataSampler(Sampler):

    def __init__(self, batch_size, X, gan, sample_cache_nr='all', force_equal_batch_size=True):
        super().__init__(batch_size, X, gan, sample_cache_nr, force_equal_batch_size)
        self.dataset, self.dataset_iterator = self._create_dataset()

    def _create_dataset(self, repeat=1):
        samples = self._sample_x(self.sample_cache_nr)
        dataset = tf.data.Dataset.from_tensor_slices(samples).repeat(repeat).batch(batch_size=self.batch_size,
                                                                                   drop_remainder=self.force_equal_batch_size)
        dataset_iterator = iter(dataset)
        return dataset, dataset_iterator

    @abstractmethod
    def _sample_x(self, nr=1):
        ...

    def _take_or_reconstruct(self):
        sample = next(self.dataset_iterator, None)
        if sample is None:
            self._create_dataset()
            sample = next(self.dataset_iterator)
        return sample

    def sample_batch(self):
        return self._take_or_reconstruct()

    'Resamples nr samples. Does not make use of the cached samples'

    def sample(self, nr=1):
        return self._sample_x(nr)


class ShuffleSampler(DataSampler):

    def __init__(self, batch_size, X, gan, sample_cache_nr='all', force_equal_batch_size=True):
        super().__init__(batch_size, X, gan, sample_cache_nr, force_equal_batch_size)

    def _create_dataset(self, repeat=None):
        samples = self._sample_x(self.sample_cache_nr)
        dataset = tf.data.Dataset.from_tensor_slices(samples).repeat(repeat).shuffle(self.sample_cache_nr).batch(
            batch_size=self.batch_size,
            drop_remainder=self.force_equal_batch_size)
        dataset_iterator = iter(dataset)
        return dataset, dataset_iterator

    def _sample_x(self, nr=1):
        if nr == self.X.shape[0]:
            return self.X
        else:
            return self.X[:nr]

class NumpyUniformSampler(Sampler):

    def __init__(self, batch_size, X, gan, sample_cache_nr='all', force_equal_batch_size=True):
        super().__init__(batch_size, X, gan, sample_cache_nr, force_equal_batch_size)
        self.np_generator = default_rng()
        self.dataset_iterator = uniform_sampler_generator(self.batch_size,self.X, self.sample_cache_nr, np_generator=self.np_generator)

    def sample_batch(self):
        b = next(self.dataset_iterator, None)
        if b is None:
            self.dataset_iterator = uniform_sampler_generator(self.batch_size,self.X, self.sample_cache_nr)
            b = next(self.dataset_iterator)
        return b
