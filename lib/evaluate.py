import itertools
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
import tensorflow_gan as tfgan
import os
from lib.data import sample_batch_uniform
from lib import models, data
import math


def evaluate_nr_modes_and_high_quality_samples(means, xx, std):
    l2_store = []
    for x_ in xx:
        l2_store.append([np.sum((x_ - i) ** 2) for i in means])

    mode = np.argmin(l2_store, 1).flatten().tolist()
    dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
    mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i]) <= 3 * std]

    nr_modes_captured = len(collections.Counter(mode_counter))
    nr = np.sum(list(collections.Counter(mode_counter).values()))
    percentage_within_3std = nr / xx.shape[0]
    return nr_modes_captured, percentage_within_3std


def evaluate_grid(xx):
    means = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2), range(-4, 5, 2))],
                     dtype=np.float32)
    return evaluate_nr_modes_and_high_quality_samples(means, xx, 0.05)


def evaluate_ring(xx):
    radius = 2.5
    n_mixture = 8
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    means = np.asarray([np.asarray([xi, yi]) for xi, yi in zip(xs.ravel(), ys.ravel())])
    return evaluate_nr_modes_and_high_quality_samples(means, xx, 0.01)


def IvOLossTotal(generator, targets, batchSize, noise_dim):
    def IvOLoss(noise):
        noise = tf.reshape(noise, [batchSize, noise_dim])
        loss, grad = tfp.math.value_and_gradient(lambda noise: tf.nn.l2_loss(generator(noise, training=False) - targets), noise)
        grad = tf.reshape(noise, [batchSize * noise_dim])
        return loss, grad

    return IvOLoss

def get_inference_via_optimization(gan,X, batch_size,sample_nr=10000, num_of_random_restarts=25, write_dir=None):      
    iterations = np.round(sample_nr/batch_size).astype(int)
    if iterations < 1:
        iterations = 1
    IvO = np.zeros(iterations)
    for i in range(iterations):
      targets = sample_batch_uniform(batch_size, X) 
      IvO[i] = get_single_inference_via_optimization(gan,targets,batch_size, num_of_random_restarts, write_dir)
    IvO_mean = np.mean(IvO)
    return IvO_mean


def get_single_inference_via_optimization(gan, targets, batch_size, num_of_random_restarts=25, write_dir=None):
    # decide on batch_size images (picked randomly) to try to reconstruct
    generator = gan.generator
    z_sampler = gan.z_sampler
    noise_dim = generator.input_shape[1]
    if write_dir is not None:
        save_loc = os.path.join(write_dir, 'IvO/Images/targets')
        np.save(save_loc, targets[:, :, :, 0])

    # run the optimization from different initializations
    imSize = tf.shape(targets[0,:,:,:])
    results_images = np.ones((num_of_random_restarts,batch_size,imSize[0], imSize[1], imSize[2]))
    results_errors = np.ones((num_of_random_restarts, batch_size))

    for i in range(num_of_random_restarts):
        # randomly initialize z
        start = z_sampler([batch_size, noise_dim])
        start = tf.reshape(start, [batch_size * noise_dim])

        # Loss function
        IvOLoss = IvOLossTotal(generator, targets, batch_size, noise_dim)

        # Optimize
        optim_results = tfp.optimizer.lbfgs_minimize(IvOLoss, initial_position=start, num_correction_pairs=100, max_iterations=500)
        noise_optimal = optim_results.position
        noise_optimal = tf.reshape(noise_optimal, [batch_size, noise_dim])

        # Results
        generated_samples = generator(noise_optimal, training=False)
        generated_samples_mse = tf.reduce_mean(tf.square(generated_samples - targets), axis=[1, 2, 3])
        results_images[i, :, :, :,:] = generated_samples
        results_errors[i, :] = generated_samples_mse

        # tf.print('Optimization Run Finished')

    # select the best out of all random restarts
    best_images = np.ones(tf.shape(results_images[0,:,:,:,:]))
    best_images_errors = np.ones(batch_size)
    for image_index in range(batch_size):
        best_img = results_images[0, image_index, :, :,:]
        best_img_error = results_errors[0, image_index]
        for indep_run_index in range(1, num_of_random_restarts):
            if best_img_error > results_errors[indep_run_index, image_index]:
                best_img_error = results_errors[indep_run_index, image_index]
                best_img = results_images[indep_run_index, image_index, :, :,:]
        best_images[image_index, :, :, :] = best_img
        best_images_errors[image_index] = best_img_error

    if write_dir is not None:
        save_loc1 = os.path.join(write_dir, 'IvO/Images/IvOImages')
        save_loc2 = os.path.join(write_dir, 'IvO/Results/errors')
        np.save(save_loc1, best_images)
        np.save(save_loc2, best_images_errors)

    return np.mean(best_images_errors)


def sample_KL_divergence(a, b, n_unique):
    a_probs = np.zeros(n_unique)
    b_probs = np.zeros(n_unique)

    for y in a:
        a_probs[y] = a_probs[y] + 1
    for y in b:
        b_probs[y] = b_probs[y] + 1

    a_probs = a_probs / (len(a))
    b_probs = b_probs / (len(b))
    a_probs = tf.clip_by_value(a_probs, tf.keras.backend.epsilon(), 1) # prevents NaN or inf because of 0 prob
    b_probs = tf.clip_by_value(b_probs, tf.keras.backend.epsilon(), 1) # prevents NaN or inf because of 0 prob

    X = tfp.distributions.Categorical(probs=a_probs)
    Y = tfp.distributions.Categorical(probs=b_probs)
    return tfp.distributions.kl_divergence(X, Y)


def count_stacked_mnist_modes_with_KL(gan, true_labels, cnn_mnist_model_location='data/cnn_model_checkpoint.hdf5',
                                      sample_nr=26000, batch_size=32):
    samples = gan.sample_generator(sample_nr, batch_size=batch_size)
    cnn = models.mnist_cnn()
    cnn.load_weights(cnn_mnist_model_location)

    channel1 = cnn.predict(np.expand_dims(samples[:, :, :, 0], -1), batch_size=batch_size)
    channel2 = cnn.predict(np.expand_dims(samples[:, :, :, 1], -1), batch_size=batch_size)
    channel3 = cnn.predict(np.expand_dims(samples[:, :, :, 2], -1), batch_size=batch_size)

    y_pred = []
    for i in range(channel1.shape[0]):
        y_pred.append(np.argmax(channel1[i]) * 100 + np.argmax(channel2[i]) * 10 + np.argmax(channel3[i]))
    unique_y_samples = np.unique(y_pred)

    true_labels = [np.argmax(label) for label in true_labels]

    return unique_y_samples.shape[0], sample_KL_divergence(y_pred, true_labels, n_unique=1000)


def _resize_images_batch_wise(images, new_size, batch_size, method=tf.image.ResizeMethod.BILINEAR):
    resized_images = np.zeros((images.shape[0], new_size[0], new_size[1], images.shape[3]), dtype='float32')
    curr_index = 0
    while curr_index < images.shape[0]:
        if curr_index + batch_size > images.shape[0]:
            curr_generated_batch = images[curr_index:]
            resized_images[curr_index:] = tf.image.resize(
                curr_generated_batch, [new_size[0], new_size[1]], method=method)
        else:
            curr_batch = images[curr_index:curr_index + batch_size]
            resized_images[curr_index:curr_index + batch_size] = tf.image.resize(
                curr_batch, [new_size[0], new_size[1]], method=method)
        curr_index = curr_index + batch_size
    return resized_images


# Evaluates how well the inceptionv3 network is able to classify the generated images
def inception_score(generated_images, batch_size=32, resize_method = None):
    if resize_method is not None:
        size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        generated_images = _resize_images_batch_wise(generated_images, (size, size), batch_size, method = resize_method)
    num_batches = math.ceil(generated_images.shape[0] / batch_size)
    return tfgan.eval.inception_score(generated_images, num_batches=num_batches)


# In addition to the inception score, a comparison is also made to real images
def frechet_inception_distance(generated_images, x, batch_size=32, resize_method = None):
    if resize_method is not None:
        size = tfgan.eval.INCEPTION_DEFAULT_IMAGE_SIZE
        generated_images = _resize_images_batch_wise(generated_images, (size, size), batch_size, method=resize_method)
        x = _resize_images_batch_wise(x, (size, size), batch_size, method=resize_method)
    num_batches = math.ceil(generated_images.shape[0] / batch_size)
    return tfgan.eval.inception_metrics.frechet_inception_distance(x,
                                                                   generated_images,
                                                                   num_batches=num_batches)



# the Wasserstein distance between real and fake images.
def sliced_wasserstein_distance(generated_images, x, sample_nr=10000):
    generated_images = tf.convert_to_tensor(generated_images)
    x = tf.convert_to_tensor(x[:sample_nr])
    distance = _sliced_wasserstein_distance(x, generated_images)
    distances_fake = []
    for d in distance:
        distances_fake.append(d[1])
    return distances_fake


class InceptionScorer:

    def __init__(self):
        self.tfhub_inception = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
        self.inception_layer = hub.KerasLayer(self.tfhub_inception)
        self.cached_fid_real_outputs = None

    def __kl_divergence(self, p, p_logits, q):
        x = p * (tf.nn.log_softmax(p_logits) - tf.math.log(q))
        return tf.reduce_sum(x, axis=1)

    def __get_outputs(self, images, batch_size, final_layer):
      all_outputs = []
      for batch in data.batch_generator(images, batch_size=batch_size):
          out = self.inception_layer(batch)
          out = out[final_layer]
          if final_layer =='pool_3':
            out = tf.squeeze(out)
          all_outputs.append(out)
      return all_outputs

    def inception_score(self,images, batch_size=1):
        all_logits = self.__get_outputs(images, batch_size, 'logits')
        logits = tf.concat(all_logits, axis=0)
        p = tf.nn.softmax(logits)
        q = tf.reduce_mean(input_tensor=p, axis=0)
        kl = self.__kl_divergence(p, logits, q)
        return tf.exp(tf.reduce_mean(kl))

    def frechet_inception_distance(self, real_images, fake_images, batch_size=1, cache_real_output = False):
        if cache_real_output:
            if self.cached_fid_real_outputs is None:
                self.cached_fid_real_outputs = self.__get_outputs(real_images, batch_size, 'pool_3')
            all_outputs1 = self.cached_fid_real_outputs
        else:
            all_outputs1 = self.__get_outputs(real_images, batch_size, 'pool_3')
        all_outputs2 = self.__get_outputs(fake_images, batch_size, 'pool_3')
        return tfgan.eval.frechet_classifier_distance_from_activations(all_outputs1,all_outputs2)


############################################ CODE SWD ###########################################################

# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Sliced Wasserstein Distance.
Proposed in https://arxiv.org/abs/1710.10196 and the official Theano
implementation that we used as reference can be found here:
https://github.com/tkarras/progressive_growing_of_gans
Note: this is not an exact distance but an approximation through random
projections.
"""


from tensorflow_gan.python import contrib_utils as contrib

__all__ = ['sliced_wasserstein_distance']

_GAUSSIAN_FILTER = np.float32([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]]).reshape([5, 5, 1, 1]) / 256.0


def _to_float(tensor):
  return tf.cast(tensor, tf.float32)


def laplacian_pyramid(batch, num_levels):
  """Compute a Laplacian pyramid.
  Args:
      batch: (tensor) The batch of images (batch, height, width, channels).
      num_levels: (int) Desired number of hierarchical levels.
  Returns:
      List of tensors from the highest to lowest resolution.
  """
  gaussian_filter = tf.constant(_GAUSSIAN_FILTER)

  def spatial_conv(batch, gain):
    """Custom conv2d."""
    s = tf.shape(input=batch)
    padded = tf.pad(
        tensor=batch, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='REFLECT')
    xt = tf.transpose(a=padded, perm=[0, 3, 1, 2])
    xt = tf.reshape(xt, [s[0] * s[3], s[1] + 4, s[2] + 4, 1])
    conv_out = tf.nn.conv2d(
        input=xt,
        filters=gaussian_filter * gain,
        strides=[1] * 4,
        padding='VALID')
    conv_xt = tf.reshape(conv_out, [s[0], s[3], s[1], s[2]])
    conv_xt = tf.transpose(a=conv_xt, perm=[0, 2, 3, 1])
    return conv_xt

  def pyr_down(batch):  # matches cv2.pyrDown()
    return spatial_conv(batch, 1)[:, ::2, ::2]

  def pyr_up(batch):  # matches cv2.pyrUp()
    s = tf.shape(input=batch)
    zeros = tf.zeros([3 * s[0], s[1], s[2], s[3]])
    res = tf.concat([batch, zeros], 0)
    res = contrib.batch_to_space(
        input=res, crops=[[0, 0], [0, 0]], block_shape=2)
    res = spatial_conv(res, 4)
    return res

  pyramid = [_to_float(batch)]
  for _ in range(1, num_levels):
    pyramid.append(pyr_down(pyramid[-1]))
    pyramid[-2] -= pyr_up(pyramid[-1])
  return pyramid


def _batch_to_patches(batch, patches_per_image, patch_size):
  """Extract patches from a batch.
  Args:
      batch: (tensor) The batch of images (batch, height, width, channels).
      patches_per_image: (int) Number of patches to extract per image.
      patch_size: (int) Size of the patches (size, size, channels) to extract.
  Returns:
      Tensor (batch*patches_per_image, patch_size, patch_size, channels) of
      patches.
  """

  def py_func_random_patches(batch):
    """Numpy wrapper."""
    batch_size, height, width, channels = batch.shape
    patch_count = patches_per_image * batch_size
    hs = patch_size // 2
    # Randomly pick patches.
    patch_id, y, x, chan = np.ogrid[0:patch_count, -hs:hs + 1, -hs:hs + 1, 0:3]
    img_id = patch_id // patches_per_image
    # pylint: disable=g-no-augmented-assignment
    # Need explicit addition for broadcast to work properly.
    y = y + np.random.randint(hs, height - hs, size=(patch_count, 1, 1, 1))
    x = x + np.random.randint(hs, width - hs, size=(patch_count, 1, 1, 1))
    # pylint: enable=g-no-augmented-assignment
    idx = ((img_id * height + y) * width + x) * channels + chan
    patches = batch.flat[idx]
    return patches

  patches = tf.compat.v1.py_func(
      py_func_random_patches, [batch], batch.dtype, stateful=False)
  return patches


def _normalize_patches(patches):
  """Normalize patches by their mean and standard deviation.
  Args:
      patches: (tensor) The batch of patches (batch, size, size, channels).
  Returns:
      Tensor (batch, size, size, channels) of the normalized patches.
  """
  patches = tf.concat(patches, 0)
  mean, variance = tf.nn.moments(x=patches, axes=[1, 2, 3], keepdims=True)
  patches = (patches - mean) / (tf.sqrt(variance)  + 1e-12)
  return tf.reshape(patches, [tf.shape(input=patches)[0], -1])


def _sort_rows(matrix, num_rows):
  """Sort matrix rows by the last column.
  Args:
      matrix: a matrix of values (row,col).
      num_rows: (int) number of sorted rows to return from the matrix.
  Returns:
      Tensor (num_rows, col) of the sorted matrix top K rows.
  """
  tmatrix = tf.transpose(a=matrix, perm=[1, 0])
  sorted_tmatrix = tf.nn.top_k(tmatrix, num_rows)[0]
  return tf.transpose(a=sorted_tmatrix, perm=[1, 0])


def _sliced_wasserstein(a, b, random_sampling_count, random_projection_dim):
  """Compute the approximate sliced Wasserstein distance.
  Args:
      a: (matrix) Distribution "a" of samples (row, col).
      b: (matrix) Distribution "b" of samples (row, col).
      random_sampling_count: (int) Number of random projections to average.
      random_projection_dim: (int) Dimension of the random projection space.
  Returns:
      Float containing the approximate distance between "a" and "b".
  """
  s = tf.shape(input=a)
  means = []
  for _ in range(random_sampling_count):
    # Random projection matrix.
    proj = tf.random.normal([tf.shape(input=a)[1], random_projection_dim])
    proj *= tf.math.rsqrt(
        tf.reduce_sum(input_tensor=tf.square(proj), axis=0, keepdims=True))
    # Project both distributions and sort them.
    proj_a = tf.matmul(a, proj)
    proj_b = tf.matmul(b, proj)
    proj_a = _sort_rows(proj_a, s[0])
    proj_b = _sort_rows(proj_b, s[0])
    # Pairwise Wasserstein distance.
    wdist = tf.reduce_mean(input_tensor=tf.abs(proj_a - proj_b))
    means.append(wdist)
  return tf.reduce_mean(input_tensor=means)


def _sliced_wasserstein_svd(a, b):
  """Compute the approximate sliced Wasserstein distance using an SVD.
  This is not part of the paper, it's a variant with possibly more accurate
  measure.
  Args:
      a: (matrix) Distribution "a" of samples (row, col).
      b: (matrix) Distribution "b" of samples (row, col).
  Returns:
      Float containing the approximate distance between "a" and "b".
  """
  s = tf.shape(input=a)
  # Random projection matrix.
  sig, u = tf.linalg.svd(tf.concat([a, b], 0))[:2]
  proj_a, proj_b = tf.split(u * sig, 2, axis=0)
  proj_a = _sort_rows(proj_a[:, ::-1], s[0])
  proj_b = _sort_rows(proj_b[:, ::-1], s[0])
  # Pairwise Wasserstein distance.
  wdist = tf.reduce_mean(input_tensor=tf.abs(proj_a - proj_b))
  return wdist


def _sliced_wasserstein_distance(real_images,
                                fake_images,
                                resolution_min=16,
                                patches_per_image=64,
                                patch_size=7,
                                random_sampling_count=1,
                                random_projection_dim=7 * 7 * 3,
                                use_svd=False):
  """Compute the Wasserstein distance between two distributions of images.
  Note that measure vary with the number of images. Use 8192 images to get
  numbers comparable to the ones in the original paper.
  Args:
      real_images: (tensor) Real images (batch, height, width, channels).
      fake_images: (tensor) Fake images (batch, height, width, channels).
      resolution_min: (int) Minimum resolution for the Laplacian pyramid.
      patches_per_image: (int) Number of patches to extract per image per
        Laplacian level.
      patch_size: (int) Width of a square patch.
      random_sampling_count: (int) Number of random projections to average.
      random_projection_dim: (int) Dimension of the random projection space.
      use_svd: experimental method to compute a more accurate distance.
  Returns:
      List of tuples (distance_real, distance_fake) for each level of the
      Laplacian pyramid from the highest resolution to the lowest.
        distance_real is the Wasserstein distance between real images
        distance_fake is the Wasserstein distance between real and fake images.
  Raises:
      ValueError: If the inputs shapes are incorrect. Input tensor dimensions
      (batch, height, width, channels) are expected to be known at graph
      construction time. In addition height and width must be the same and the
      number of colors should be exactly 3. Real and fake images must have the
      same size.
  """
  height = real_images.shape[1]
  real_images.shape.assert_is_compatible_with([None, None, height, 3])
  fake_images.shape.assert_is_compatible_with(real_images.shape)

  # Select resolutions.
  resolution_full = int(height)
  resolution_min = min(resolution_min, resolution_full)
  resolution_max = resolution_full
  # Base loss of detail.
  resolutions = [
      2**i for i in range(
          int(np.log2(resolution_max)),
          int(np.log2(resolution_min)) - 1, -1)
  ]

  # Gather patches for each level of the Laplacian pyramids.
  patches_real, patches_fake, patches_test = (
      [[] for _ in resolutions] for _ in range(3))
  for lod, level in enumerate(laplacian_pyramid(real_images, len(resolutions))):
    patches_real[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))
    patches_test[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))

  for lod, level in enumerate(laplacian_pyramid(fake_images, len(resolutions))):
    patches_fake[lod].append(
        _batch_to_patches(level, patches_per_image, patch_size))

  for lod in range(len(resolutions)):
    for patches in [patches_real, patches_test, patches_fake]:
      patches[lod] = _normalize_patches(patches[lod])

  # Evaluate scores.
  scores = []
  for lod in range(len(resolutions)):
    if not use_svd:
      scores.append(
          (_sliced_wasserstein(patches_real[lod], patches_test[lod],
                               random_sampling_count, random_projection_dim),
           _sliced_wasserstein(patches_real[lod], patches_fake[lod],
                               random_sampling_count, random_projection_dim)))
    else:
      scores.append(
          (_sliced_wasserstein_svd(patches_real[lod], patches_test[lod]),
           _sliced_wasserstein_svd(patches_real[lod], patches_fake[lod])))
  return scores



