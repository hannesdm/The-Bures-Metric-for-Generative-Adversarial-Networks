import os

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from lib import losses, data
from lib.logging import TensorboardLogger, save_samples

default_adam_beta1 = 0.5
default_adam_learning_rate = 1e-3
default_z_sampler = tf.random.normal
default_x_sampler = 'npuniform'


class GAN:

    def __init__(self, generator, discriminator, generator_optimizer=None,
                 discriminator_optimizer=None, generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, adaptive_LR='fixed'):
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name
        self.generator = generator
        if type(discriminator) is tuple:
            self.discriminator = discriminator[0]
            self.discriminator_feature_map = discriminator[1]
        else:
            self.discriminator = discriminator
        if generator_optimizer is None:
            generator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                           learning_rate=default_adam_learning_rate)
        self.generator_optimizer = generator_optimizer
        if discriminator_optimizer is None:
            discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                               learning_rate=default_adam_learning_rate)
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.z_dim = generator.input_shape[1]
        self.z_sampler = z_sampler
        self.x_sampler_method = x_sampler_method
        self.x_sampler_args = x_sampler_args
        self.x_sampler = None

    def update_discriminator(self, noise, real_batch):
        with tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=False)
            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)
            disc_loss = self.discriminator_loss(real_y, fake_y)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss

    def update_generator(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_batch = self.generator(noise, training=True)
            fake_y = self.discriminator(generated_batch, training=False)
            gen_loss = self.generator_loss(fake_y)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def sample_generator(self, nr, batch_size=None):
        noise = self.z_sampler([nr, self.z_dim])
        if batch_size is None:
            return self.generator(noise, training=False)
        else:
            return self.generator.predict(noise, batch_size=batch_size)

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=True)

            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)

            gen_loss = self.generator_loss(fake_y)
            disc_loss = self.discriminator_loss(real_y, fake_y)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def log_losses(self, logger, step, losses):
        logger.write_group([losses[0], losses[1]], group_name="losses", i=step,
                           names=['Generator_loss', 'Discriminator_loss'])

    def save_generator(self, logdir='out', name=None):
        if name is None:
            name = self.name
        save_path = os.path.join(logdir,'models', name)
        self.generator.save_weights(save_path)

    def load_generator(self, logdir='out', name=None):
        if name is None:
            name = self.name
        path = os.path.join(logdir, 'models', name)
        self.generator.load_weights(path)

    def _sample_x_batches(self):
        return self.x_sampler.sample(1)

    def train(self, X, batch_size=32, steps=50000, log_losses=False, logdir='out', save_samples_every=-1):
        plot = save_samples_every > 0
        logdir = os.path.join(logdir, self.name)

        if log_losses:
            logger = TensorboardLogger(logdir)

        if self.x_sampler is None:
            self.x_sampler = data.Sampler.create(self.x_sampler_method, batch_size, X, self, self.x_sampler_args)

        with tqdm(total=steps) as pbar:
            for step in range(steps):
                x_batches = self._sample_x_batches()
                losses = self.train_step(batch_size, x_batches=x_batches)
                # save intermediate images
                if plot and (step % save_samples_every == 0):
                    save_samples(self.z_sampler, self.generator, X=X,
                                 root_dir=os.path.join(logdir, 'plots/'), filename='{}_{}'.format(self.name, step))
                # log tensorboard
                if log_losses:
                    self.log_losses(logger, step, losses)
                pbar.update()

        if plot:  # plot final result
            save_samples(self.z_sampler, self.generator, X=X,
                         root_dir=os.path.join(logdir, 'plots/'), filename='{}_{}'.format(self.name, step))

    def train_epochs(self, X, batch_size=32, epochs=10, log_losses=False, logdir='out', save_samples_every_epoch=False):
        logdir = os.path.join(logdir, self.name)

        if log_losses:
            logger = TensorboardLogger(logdir)

        # only supports the default epoch batch sampler
        self.x_sampler = data.Sampler.create("epoch", batch_size, X, None, None)

        batch_count = X.shape[0] // batch_size
        with tqdm(total=epochs) as pbar:
            for step in range(epochs):
                for _ in range(batch_count):
                    x_batches = self._sample_x_batches()
                    losses = self.train_step(batch_size, x_batches=x_batches)
                    # log tensorboard
                    if log_losses:
                        self.log_losses(logger, step, losses)
                        # save intermediate images

                if save_samples_every_epoch:
                    save_samples(self.z_sampler, self.generator, X=X,
                                 root_dir=os.path.join(logdir, 'plots/'),
                                 filename='{}_{}'.format(self.name, step))
                pbar.update()

        if save_samples_every_epoch:  # plot final result
            save_samples(self.z_sampler, self.generator, X=X,
                         root_dir=os.path.join(logdir, 'plots/'), filename='{}_{}'.format(self.name, step))


class DiverseGAN(GAN):

    def __init__(self, generator, discriminator_with_feature_map, generator_optimizer=None,
                 discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss,
                 z_sampler=default_z_sampler, x_sampler_method=default_x_sampler, x_sampler_args=None, name=None,
                 diversity_loss_func=None):
        super().__init__(generator, discriminator_with_feature_map, generator_optimizer, discriminator_optimizer,
                         generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method=x_sampler_method,
                         x_sampler_args=x_sampler_args, name=name)
        self.diversity_loss_func = diversity_loss_func
        if diversity_loss_func is None or diversity_loss_func is None:
            raise AssertionError("DiverseGAN requires a discriminator feature map and a diversity loss function.")

    def log_losses(self, logger, step, losses):
        logger.write_group([losses[0], losses[1], losses[2]], group_name="losses", i=step,
                           names=['Generator_loss', 'Discriminator_loss', 'Diversity_loss'])

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=True)

            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)

            phi_real = self.discriminator_feature_map(real_batch, training=True)
            phi_fake = self.discriminator_feature_map(generated_batch, training=True)

            diversity_loss = self.diversity_loss_func(phi_fake, phi_real)
            gen_cross_entropy = self.generator_loss(fake_y)
            gen_cross_entropy = tf.cast(gen_cross_entropy, tf.float64)
            gen_loss = 0.5 * (gen_cross_entropy + diversity_loss)
            disc_loss = self.discriminator_loss(real_y, fake_y)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_cross_entropy, disc_loss, diversity_loss


class GDPPGAN(DiverseGAN):

    def __init__(self, generator, discriminator_with_feature_map, generator_optimizer=None,
                 discriminator_optimizer=None, generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, diversity_loss_func=None):
        diversity_loss_func = losses.gdpp_diversity_loss
        super().__init__(generator, discriminator_with_feature_map, generator_optimizer, discriminator_optimizer,
                         generator_loss, discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name,
                         diversity_loss_func)


class BuresGAN(DiverseGAN):

    def __init__(self, generator, discriminator_with_feature_map, generator_optimizer=None,
                 discriminator_optimizer=None, generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, diversity_loss_func=None,
                 dual=False):
        if diversity_loss_func is None:
            if dual:
                diversity_loss_func = losses.wasserstein_bures_kernel
            else:
                diversity_loss_func = losses.wasserstein_bures_covariance
        super().__init__(generator, discriminator_with_feature_map, generator_optimizer, discriminator_optimizer,
                         generator_loss, discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name,
                         diversity_loss_func)


class AlternatingBuresGAN(BuresGAN):

    def _sample_x_batches(self):
        return self.x_sampler.sample(2)

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]

        with tf.GradientTape() as gen_tape1:
            generated_batch = self.generator(noise, training=True)
            phi_real = self.discriminator_feature_map(real_batch, training=True)
            phi_fake = self.discriminator_feature_map(generated_batch, training=True)
            diversity_loss = self.diversity_loss_func(phi_fake, phi_real)
        gradients_of_generator = gen_tape1.gradient(diversity_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[1]

        with tf.GradientTape() as gen_tape2, tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=True)

            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)

            gen_cross_entropy = self.generator_loss(fake_y)
            disc_loss = self.discriminator_loss(real_y, fake_y)

        gradients_of_generator = gen_tape2.gradient(gen_cross_entropy, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_cross_entropy, disc_loss, diversity_loss


# First variant of MDGAN described in section 3.2 'Mode Regularizer' of the paper.
# No mention is made of the encoder optimizer (and whether its shared or not) so we use the a separate
# default adam optimizer unless otherwise specified as a parameter.
# The default lambda parameters are those found in section 4.1.1 of the paper
class MDGANv1(GAN):

    def __init__(self, generator, discriminator1, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, encoder=None, encoder_optimizer=None
                 , lam1=0.2, lam2=0.4):
        super().__init__(generator, discriminator1, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name)
        if encoder is None:
            raise AssertionError("MDGAN requires an encoder model.")
        self.encoder = encoder
        if encoder_optimizer is None:
            encoder_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                               learning_rate=default_adam_learning_rate)
        self.encoder_optimizer = encoder_optimizer
        self.lam1 = lam1
        self.lam2 = lam2

    def _sample_x_batches(self):
        return self.x_sampler.sample(2)

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]

        disc_loss = self.update_discriminator(noise=noise, real_batch=real_batch)

        with tf.GradientTape() as enc_tape:
            encoded_noise = self.encoder(real_batch)
            G_e = self.generator(encoded_noise)
            D_G_e = self.discriminator(G_e)
            enc_loss = self.lam1 * tf.reduce_sum(
                tf.losses.mse(real_batch, G_e)) + self.lam2 * losses.cross_entropy_from_logits(tf.zeros_like(D_G_e), D_G_e)

        gradients_of_encoder = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        self.encoder_optimizer.apply_gradients(
            zip(gradients_of_encoder, self.encoder.trainable_variables))

        with tf.GradientTape() as gen_tape:
            generated_batch = self.generator(noise)
            fake_y = self.discriminator(generated_batch)
            gen_cross_entropy = self.generator_loss(fake_y)
            encoded_noise = self.encoder(real_batch)
            G_e = self.generator(encoded_noise)
            D_G_e = self.discriminator(G_e)
            enc_loss = self.lam1 * tf.reduce_sum(tf.losses.mse(real_batch, G_e)) + self.lam2 * losses.cross_entropy_from_logits(tf.zeros_like(D_G_e), D_G_e)
            gen_loss = gen_cross_entropy + enc_loss


        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_cross_entropy, disc_loss

# The 2nd variant of MDGAN described in section 3.3 and appendix A. According to the authors,
# this variant performs better for some large scale datasets since it's easier to tune.
# In the training procedure in appendix A, not mention is made of updating the encoder, we assume this is updated
# together with the first generator update in their 'manifold step'.
class MDGANv2(GAN):

    def __init__(self, generator, discriminator1, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, discriminator2=None,
                 discriminator2_optimizer=None, encoder=None
                 , lam=1e-2):
        super().__init__(generator, discriminator1, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name)
        if discriminator2 is None:
            discriminator2 = tf.keras.models.clone_model(self.discriminator)
        self.discriminator2 = discriminator2
        if discriminator2_optimizer is None:
            discriminator2_optimizer = tf.keras.optimizers.Adam(beta_1=default_adam_beta1,
                                                                learning_rate=default_adam_learning_rate)
            self.discriminator2_optimizer = discriminator2_optimizer
        if encoder is None:
            raise AssertionError("MDGAN requires an encoder model.")
        self.encoder = encoder
        self.lam = lam

    def _sample_x_batches(self):
        return self.x_sampler.sample(2)

    @tf.function
    def train_step(self, batch_size, x_batches):
        real_batch = x_batches[0]

        # step 1
        with tf.GradientTape() as disc1_tape:
            enc_real_batch = self.encoder(real_batch)
            gen_enc_real_batch = self.generator(enc_real_batch)
            d1_real_batch = self.discriminator(real_batch)
            d1_gen_enc = self.discriminator(gen_enc_real_batch)
            loss_D1 = self.discriminator_loss(d1_real_batch, d1_gen_enc)
        gradients_of_discriminator1 = disc1_tape.gradient(loss_D1, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator1, self.discriminator.trainable_variables))
        # step 2
        with tf.GradientTape() as gen_tape:
            loss_G = self.generator_loss(self.discriminator(self.generator(self.encoder(real_batch))))
            G_sample_reg = self.generator(self.encoder(real_batch))
            mse = tf.reduce_sum((real_batch - G_sample_reg) ** 2, 1)
            loss_G = self.lam * loss_G + tf.reduce_mean(mse)
        gradients_of_generator = gen_tape.gradient(loss_G,
                                                   self.generator.trainable_variables + self.encoder.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables + self.encoder.trainable_variables))
        # step 3
        real_batch = x_batches[1]
        noise = self.z_sampler([batch_size, self.z_dim])
        with tf.GradientTape() as disc2_tape:
            loss_D2 = self.discriminator_loss(self.discriminator2(self.generator(self.encoder(real_batch))),
                                              self.discriminator2(self.generator(noise)))
        gradients_of_discriminator2 = disc2_tape.gradient(loss_D2, self.discriminator2.trainable_variables)
        self.discriminator2_optimizer.apply_gradients(
            zip(gradients_of_discriminator2, self.discriminator2.trainable_variables))
        # step 4
        with tf.GradientTape() as gen2_tape:
            loss_G2 = self.generator_loss(self.discriminator2(self.generator(noise)))
        gradients_of_generator2 = gen2_tape.gradient(loss_G2, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator2, self.generator.trainable_variables))

        return loss_G2, loss_D2


class UnrolledGAN(GAN):

    def __init__(self, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, unrolling_steps=5):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name)
        self.fake_discriminator = tf.keras.models.clone_model(self.discriminator)
        self.unrolling_steps = unrolling_steps

    def update_fake_discriminator(self, noise, real_batch):
        with tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=False)
            real_y = self.fake_discriminator(real_batch, training=True)
            fake_y = self.fake_discriminator(generated_batch, training=True)
            disc_loss = self.discriminator_loss(real_y, fake_y)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.fake_discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.fake_discriminator.trainable_variables))
        return disc_loss

    def update_generator_with_fake_d(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_batch = self.generator(noise, training=True)
            fake_y = self.fake_discriminator(generated_batch, training=False)
            gen_loss = self.generator_loss(fake_y)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def _sample_x_batches(self):
        return self.x_sampler.sample(self.unrolling_steps + 1)

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]

        disc_loss = self.update_discriminator(noise, real_batch)
        # Unroll optimization of the discriminator
        if self.unrolling_steps > 0:
            # undo previous update of fake D and set the weights to be the current D
            for fake_m, real_m in zip(self.fake_discriminator.variables, self.discriminator.variables):
                fake_m.assign(real_m)  # copy variables
            for i in range(self.unrolling_steps - 1):
                noise = self.z_sampler([batch_size, self.z_dim])
                real_batch = x_batches[i + 1]

                disc_loss = self.update_fake_discriminator(noise, real_batch)
            gen_loss = self.update_generator_with_fake_d(noise)
        else:
            gen_loss = self.update_generator(noise)
        return gen_loss, disc_loss


class VEEGAN(GAN):

    def __init__(self, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, inverse_generator=None):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name)
        if inverse_generator is None:
            raise AssertionError("VEEGAN requires an inverse generator model.")
        if self.discriminator.input_shape == 2 and self.discriminator.input_shape[1] != (
                generator.output_shape[1] + self.z_dim):
            raise AssertionError("VEEGAN requires a discriminator input shape of z_dim + x_dim,"
                                 " i.e. '{}' but is '{}'. Alternatively when working with images a discriminator with 2 inputs is required.".format(
                generator.output_shape[1] + self.z_dim, self.discriminator.input_shape[1]))
        self.inverse_generator = inverse_generator

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise = self.z_sampler([batch_size, self.z_dim])
        real_batch = x_batches[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_batch = self.generator(noise, training=True)

            reconstructed_z_from_real = self.inverse_generator(real_batch, training=True)
            reconstructed_z_from_fake = self.inverse_generator(generated_batch, training=True)
            if len(generated_batch.shape) > 2:  # images need a separate input to discriminator
                log_d_prior = self.discriminator([reconstructed_z_from_real, real_batch], training=True)
                log_d_posterior = self.discriminator([noise, generated_batch], training=True)
            else:
                log_d_prior = self.discriminator(tf.concat([reconstructed_z_from_real, real_batch], 1), training=True)
                log_d_posterior = self.discriminator(tf.concat([noise, generated_batch], 1), training=True)

            recon_likelihood = tf.reduce_sum(reconstructed_z_from_fake.log_prob(noise), 1)

            mean_log_d_post = tf.reduce_mean(tf.squeeze(log_d_posterior))
            mean_likelihood = tf.reduce_mean(recon_likelihood)

            gen_loss = mean_log_d_post - mean_likelihood
            disc_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) +
                tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))

        gradients_of_generator = gen_tape.gradient(gen_loss,
                                                   self.generator.trainable_variables + self.inverse_generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator,
                self.generator.trainable_variables + self.inverse_generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss


class WGAN_GP(GAN):

    def __init__(self, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, lam=10.0, n_critic=5):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name)
        self.lam = lam
        self.n_critic = n_critic

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0]] + [1] * (len(x.shape) - 1), 0.0,
                                    1.0)  # gives [batch_size, 1...,1 ] with a 1 for each dimension other than the batch size
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminator(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=list(range(1, len(x.shape)))))  # L2
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)  # expectation
        return d_regularizer

    def update_discriminator(self, noise, real_batch):
        generated_batch = self.generator(noise, training=True)
        with tf.GradientTape() as disc_tape:
            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)
            d_regularizer = self.gradient_penalty(real_batch, generated_batch)
            disc_loss = tf.reduce_mean(real_y) - tf.reduce_mean(fake_y) + self.lam * d_regularizer
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss

    def update_generator(self, noise):
        with tf.GradientTape() as gen_tape:
            generated_batch = self.generator(noise, training=True)
            fake_y = self.discriminator(generated_batch, training=True)
            gen_loss = tf.reduce_mean(fake_y)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    def _sample_x_batches(self):
        return self.x_sampler.sample(self.n_critic)

    @tf.function
    def train_step(self, batch_size, x_batches):
        disc_loss = 0.0
        for i in range(self.n_critic):
            noise = self.z_sampler([batch_size, self.z_dim])
            real_batch = x_batches[i]
            disc_loss = self.update_discriminator(noise, real_batch)

        noise = self.z_sampler([batch_size, self.z_dim])
        gen_loss = self.update_generator(noise)
        return gen_loss, disc_loss


class PACGAN(GAN):


    def __init__(self, generator, discriminator, generator_optimizer=None, discriminator_optimizer=None,
                 generator_loss=losses.cross_entropy_generator_loss,
                 discriminator_loss=losses.cross_entropy_discriminator_loss, z_sampler=default_z_sampler,
                 x_sampler_method=default_x_sampler, x_sampler_args=None, name=None, adaptive_LR='fixed', pack_nr=2):
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, generator_loss,
                         discriminator_loss, z_sampler, x_sampler_method, x_sampler_args, name, adaptive_LR)
        self.pack_nr = pack_nr

    def _sample_x_batches(self): # batch dimension is the same but pack_nr batches will be packed in the last dimension
        n_batches = self.x_sampler.sample(self.pack_nr)
        s = n_batches[0].shape
        if len(s) == 3: # add extra dim for grayscale images
            n_batches = [np.expand_dims(b,-1) for b in n_batches]
        return [np.concatenate(n_batches, axis=-1)] # concatenate in channels axis, or as a feature for 1d vector input

    @tf.function
    def train_step(self, batch_size, x_batches):
        noise_batches = [self.z_sampler([batch_size, self.z_dim]) for _ in range(self.pack_nr)]
        real_batch = x_batches[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_batches = [self.generator(noise, training=True) for noise in noise_batches]
            if len(generated_batches[0].shape) == 3:
                generated_batches = [tf.expand_dims(b,-1) for b in generated_batches]
            generated_batch = tf.concat(generated_batches, axis=-1)

            real_y = self.discriminator(real_batch, training=True)
            fake_y = self.discriminator(generated_batch, training=True)

            gen_loss = self.generator_loss(fake_y)
            disc_loss = self.discriminator_loss(real_y, fake_y)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss
