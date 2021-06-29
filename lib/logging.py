import tensorflow as tf
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from lib.data import sample_batch_uniform


class TensorboardLogger(object):
    path_regex = re.compile('[^\w\-_\. ]')

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writers = {}
        self.writer_groups = {}

    def __to_valid_name(self, s):
        return re.sub(self.path_regex, '_', s)

    def add_watch(self, tensor, name=None):
        if name is None:
            name = tensor.name
        summary_writer = tf.summary.create_file_writer(self.log_dir + "/{}".format(self.__to_valid_name(name)))
        self.writers[name] = summary_writer

    def add_watch_group(self, tensors, group_name, names=None):
        group = {}
        if names is None:
            names = [t.name for t in tensors]
        for tensor, name in zip(tensors, names):
            summary_writer = tf.summary.create_file_writer(
                self.log_dir + "/{}/{}".format(self.__to_valid_name(group_name), self.__to_valid_name(name)))
            group[name] = summary_writer
        self.writer_groups[group_name] = group

    def write(self, tensor, i, name=None):
        if name is None:
            name = tensor.name
        if name not in self.writers:
            self.add_watch(tensor)
        with self.writers[name].as_default():
            tf.summary.scalar(name=name, data=tensor, step=i)

    def write_group(self, tensors, i, group_name, names=None):
        if names is None:
            names = [t.name for t in tensors]
        if group_name not in self.writer_groups:
            self.add_watch_group(tensors, group_name, names)
        group = self.writer_groups[group_name]
        for tensor, name in zip(tensors, names):
            with group[name].as_default():
                tf.summary.scalar(name=group_name, data=tensor, step=i)

    def close(self):
        for writer in self.writers.values():
            writer.flush()
            writer.close()
        for group in self.writer_groups.values():
            for writer in group.values():
                writer.flush()
                writer.close()


def save_samples(z_sampler, generator, X=None, root_dir='plots/', filename='generated_samples'):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if generator.output_shape[1] < 3: # 2D
        sample_nr = 2500
    else:
        sample_nr = 16
    noise_dim = generator.input_shape[1]
    noise = z_sampler([sample_nr, noise_dim])
    real_samples = None
    if X is not None:
        real_samples = sample_batch_uniform(sample_nr, X)
    generated_samples = generator(noise, training=False)
    if generated_samples.shape[1] < 3:
        save_samples_2d(generated_samples, real_samples, root_dir, filename=filename)
    else:
        save_samples_as_images(generated_samples, root_dir, filename=filename)


def save_samples_2d(generated_samples, real_samples=None, root_dir='plots/', filename='generated_samples'):
    save_loc = os.path.join(root_dir, filename)
    plt.figure(figsize=(5, 5))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], edgecolor='none')
    if real_samples is not None:
        plt.scatter(real_samples[:, 0], real_samples[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.savefig(save_loc)
    plt.clf()
    plt.close()


def save_samples_as_images(samples, root_dir='plots/', filename='generated_samples'):
    save_loc = os.path.join(root_dir, filename)
    if len(samples.shape) == 3:
        np.expand_dims(samples, -1)
    gray_scale = samples.shape[3] == 1

    plt.figure(figsize=(5, 5))

    for i in range(samples.shape[0]):
        plt.subplot(4, 4, i + 1)
        if gray_scale:
            plt.imshow((samples[i, :, :, 0] + 1) * 0.5, cmap='gray')
        else:
            plt.imshow((samples[i, :, :, :] + 1) * 0.5)
        plt.axis('off')
    plt.savefig(save_loc)
    plt.close()
