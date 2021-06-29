from functools import partial

from lib.gans import *
from lib import evaluate, models, data, linalg
from lib.evaluate import InceptionScorer
from lib.gans import *
from tqdm import trange
import numpy as np
from timeit import default_timer as timer
import sys

if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU') #To not use GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    name = sys.argv[1]
    batch_size = '64'
    n_layers = '4'

    GANname = name
    name = 'StackedMNIST_Final_' + name + '_' + batch_size + '_' + n_layers
    outname = 'out_StackedMNIST_Final_' + '_' + batch_size + '_' + n_layers
    batch_size = int(batch_size)
    n_layers = int(n_layers)

    if os.path.exists(name):
        os.remove(name)


    def write_to_console_file(txt):
        with open(name + "_test", 'a') as f:
            f.write(txt + '\n')
            print(txt)


    z_dim = 100
    channels = 3
    x_dim = (28, 28, channels)
    inception_batch_size = 1  # batch size used for the inception scores
    sample_nr = 10000  # How many samples to use and evaluate
    steps = 25000
    n_rep = 10

    z_sampler = tf.random.normal
    generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5,
                                                   learning_rate=2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5,
                                                       learning_rate=2e-4)

    REGULARIZOR_sqrt = 1e-14
    WEIGHT_Bures = 1
    ITERATION_sqrt = 15
    NORMALIZATION = True
    DUAL = True

    x_sampler_args = {
        'sample_cache_nr': 'all'}  # precalculate an amount of batches - for LS this means only calculate leverage score every x batches. Set to None or a very high number when using uniform sampling


    print('--------------------------------------------------------')
    write_to_console_file('\n Regularizor sqrt= {} '.format(REGULARIZOR_sqrt))
    write_to_console_file('\n Iterations sqrt= {} '.format(ITERATION_sqrt))
    write_to_console_file('\n Weight in front of Bures= {} '.format(WEIGHT_Bures))
    write_to_console_file('\n Normalization= {} '.format(NORMALIZATION))
    write_to_console_file('\n Repetitions = {} '.format(n_rep))
    write_to_console_file('\n Training steps = {} '.format(steps))
    write_to_console_file('\n Batch size = {} '.format(batch_size))
    write_to_console_file('\n Nb of samples for evaluation = {} '.format(sample_nr))

    print('--------------------------------------------------------')

    sqrtm = lambda A: linalg.sqrt_newton_schulz(A, iterations=ITERATION_sqrt)

    div_func = lambda h1, h2: losses.wasserstein_bures_kernel(h1, h2, epsilon=REGULARIZOR_sqrt, sqrtm_func=sqrtm,
                                                              normalize=NORMALIZATION, dtype='float64',
                                                              weight=WEIGHT_Bures)

    if GANname == 'AlternatingBures':
        def create_gan_model():
            return AlternatingBuresGAN(
                models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers, return_feature_map=True),
                x_sampler_args=x_sampler_args, diversity_loss_func=div_func, z_sampler=z_sampler,
                generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'Bures':
        def create_gan_model():
            return BuresGAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                            models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers,
                                                                   return_feature_map=True),
                            diversity_loss_func=div_func, x_sampler_args=x_sampler_args, z_sampler=z_sampler,
                            generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'GAN':
        def create_gan_model():
            return GAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                       models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers),
                       x_sampler_args=x_sampler_args, z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                       discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'WGANGP':
        def create_gan_model():
            return WGAN_GP(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                           models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers,
                                                                  batchnorm=False),
                           x_sampler_args=x_sampler_args, lam=10.0, z_sampler=z_sampler,
                           generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'VEEGAN':
        def create_gan_model():
            return VEEGAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                          models.mnist_VEEGAN_discriminator_model(channels=channels, dcgan_dis_n_layers=n_layers,
                                                                  z_dim=z_dim),
                          inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim,
                                                                                                x_dim=x_dim),
                          x_sampler_args=x_sampler_args, z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                          discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'MDGANv1':
        def create_gan_model():
            return MDGANv1(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                           models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers),
                           encoder=models.mnist_dcgan_encoder_model(z_dim=z_dim, channels=channels),
                           x_sampler_args=x_sampler_args, z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'MDGANv2':
        def create_gan_model():
            return MDGANv2(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                           models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers),
                           encoder=models.mnist_dcgan_encoder_model(z_dim=z_dim, channels=channels),
                           x_sampler_args=x_sampler_args, z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'UnrolledGAN':
        def create_gan_model():
            return UnrolledGAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                               models.mnist_dcgan_discriminator_model(batchnorm=False, n_layers=n_layers,
                                                                      channels=channels), x_sampler_args=x_sampler_args,
                               z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'GDPPGAN':
        def create_gan_model():
            return GDPPGAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                           models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers,
                                                                  return_feature_map=True),
                           x_sampler_args=x_sampler_args, z_sampler=z_sampler, generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)
    elif GANname == 'PACGAN':
        def create_gan_model():
            return PACGAN(models.mnist_dcgan_generator_model_original_veegan(z_dim=z_dim, channels=channels),
                          models.mnist_dcgan_discriminator_model(channels=channels * 2, n_layers=n_layers),
                          x_sampler_args=x_sampler_args, pack_nr=2, name='PACGAN2', z_sampler=z_sampler,
                          generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer)
    else:
        raise ValueError('GAN name is unknown ' + GANname)

    mnist_results_i = {}
    print('**** STACKED MNIST ****')
    x, y = data.load_stacked_mnist()

    for i in range(n_rep):
        write_to_console_file("{}/{}".format(i + 1, n_rep))
        gan = create_gan_model()
        write_to_console_file('---- ' + GANname + ' ----')
        gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=25000, log_losses=False, logdir=outname)
        write_to_console_file('Evaluating ' + GANname + '...')
        ivo = evaluate.get_inference_via_optimization(gan, x, sample_nr=sample_nr, batch_size=batch_size)
        write_to_console_file("IvO: {}".format(ivo))
        nr_modes_captured, kl_div = evaluate.count_stacked_mnist_modes_with_KL(gan, y, batch_size=batch_size)
        write_to_console_file('Number of Modes Captured: {}'.format(nr_modes_captured))
        write_to_console_file('KL divergence: {}'.format(kl_div))

        if GANname not in mnist_results_i:
            mnist_results_i[GANname] = []
        mnist_results_i[GANname].append([nr_modes_captured, ivo, kl_div])

        gan.save_generator(outname, name=GANname + '_' + str(i))
        tf.keras.backend.clear_session()  # free memory
        del gan

    # Print results 
    for k, v in mnist_results_i.items():
        write_to_console_file('\n FINAL RESULTS MNIST' + k)
        arr = np.asarray(v)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        means = np.round(means, decimals=4)
        stds = np.round(stds, decimals=4)

        write_to_console_file('Number of Modes Captured: {}({})'.format(means[0], stds[0]))
        write_to_console_file('Ivo: = {}({})'.format(means[1], stds[1]))
        write_to_console_file('KL divergence {}({})'.format(means[2], stds[2]))
