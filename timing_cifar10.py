from lib.gans import *
from lib import evaluate, models, data, linalg
from lib.gans import *
from timeit import default_timer as timer
import numpy as np

if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU') #To not use GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    file_name = 'timings_CIFAR10'

    if os.path.exists(file_name):
        os.remove(file_name)


    def write_to_console_file(txt):
        with open(file_name, 'a') as f:
            f.write(txt + '\n')
            print(txt)


    z_dim = 100
    channels = 3
    x_dim = (32, 32, channels)
    batch_size = 256
    steps = 50
    n_rep = 5

    REGULARIZOR_sqrt = 1e-14
    WEIGHT_Bures = 1
    ITERATION_sqrt = 15
    NORMALIZATION = True
    DUAL = True


    print('--------------------------------------------------------')
    write_to_console_file('\n Regularizor sqrt= {} '.format(REGULARIZOR_sqrt))
    write_to_console_file('\n Iterations sqrt= {} '.format(ITERATION_sqrt))
    write_to_console_file('\n Weight in front of Bures= {} '.format(WEIGHT_Bures))
    write_to_console_file('\n Normalization= {} '.format(NORMALIZATION))
    write_to_console_file('\n Repetitions = {} '.format(n_rep))
    write_to_console_file('\n Epochs = {} '.format(steps))
    print('--------------------------------------------------------')

    sqrtm = lambda A: linalg.sqrt_newton_schulz(A, iterations=ITERATION_sqrt)

    div_func = lambda h1, h2: losses.wasserstein_bures_kernel(h1, h2, epsilon=REGULARIZOR_sqrt, sqrtm_func=sqrtm,
                                                              normalize=NORMALIZATION, dtype='float64',
                                                              weight=WEIGHT_Bures)


    gans = [
        lambda: AlternatingBuresGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                                       models.cifar10_dcgan_discriminator_model(return_feature_map=True),
                                       diversity_loss_func=div_func),
        lambda: BuresGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                            models.cifar10_dcgan_discriminator_model(return_feature_map=True),
                            diversity_loss_func=div_func),
        lambda: GAN(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model()),
        lambda: VEEGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                          models.cifar10_VEEGAN_discriminator_model(z_dim=z_dim),
                          inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim,
                                                                                                x_dim=x_dim)),
        lambda: MDGANv1(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(),
                           encoder=models.cifar10_dcgan_encoder_model(z_dim=z_dim)),
        lambda: MDGANv2(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(),
                           encoder=models.cifar10_dcgan_encoder_model(z_dim=z_dim)),
        lambda: UnrolledGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                               models.cifar10_dcgan_discriminator_model(batchnorm=False)),
        lambda: GDPPGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(return_feature_map=True)),
        lambda: WGAN_GP(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(batchnorm=False), lam=10.0),
        lambda: PACGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(image_input_shape=(32, 32, 3 * 2)), name='PACGAN2', pack_nr=2),
        lambda: DiverseGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(return_feature_map=True),
                           diversity_loss_func=losses.frobenius, name="Frobenius2GAN"),
    ]


    cifar_results_i = {}
    print('**** CIFAR 10 ****')
    x, y = data.load_cifar10_images()

    for gan_f in gans:
        name = None
        timings = []
        for i in range(n_rep):
            gan = gan_f()
            name = gan.name
            gan.train(x, batch_size=batch_size, steps=5, save_samples_every=-1, log_losses=False,
                      logdir='out_CIFAR10') # init TF graph
            start = timer() # actually time training
            gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=-1, log_losses=False,
                      logdir='out_CIFAR10')
            end = timer()
            time = (end - start)
            timings.append(time)

            tf.keras.backend.clear_session()  # free memory
            del gan
        means = np.mean(timings)
        stds = np.std(timings)
        means = np.round(means, decimals=4)
        stds = np.round(stds, decimals=4)
        means_per_step = np.mean(np.asarray(timings) / steps)
        stds_per_step = np.std(np.asarray(timings) / steps)
        means_per_step = np.round(means_per_step, decimals=4)
        stds_per_step = np.round(stds_per_step, decimals=4)
        write_to_console_file('\n TIMING ' + name)
        write_to_console_file('TOTAL TIME:  {}s({})'.format(means, stds))
        write_to_console_file('TIME PER STEP:  {}s({})'.format(means_per_step,stds_per_step))

