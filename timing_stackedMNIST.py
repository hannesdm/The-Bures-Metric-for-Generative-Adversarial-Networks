from lib.gans import *
from lib import evaluate, models, data, linalg
from lib.gans import *
from timeit import default_timer as timer
import numpy as np

if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU') #To not use GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    file_name = 'timings_SMNIST'

    if os.path.exists(file_name):
        os.remove(file_name)


    def write_to_console_file(txt):
        with open(file_name, 'a') as f:
            f.write(txt + '\n')
            print(txt)


    z_dim = 100
    channels = 3
    x_dim = (28, 28, channels)
    batch_size = 256
    steps = 50
    n_rep = 5

    n_layers=3

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
        lambda: AlternatingBuresGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                     models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers, return_feature_map=True) ,diversity_loss_func=div_func)  ,
        lambda: BuresGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                     models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers, return_feature_map=True),
                     diversity_loss_func=div_func),
        lambda: GAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers)),
        lambda: VEEGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                   models.mnist_VEEGAN_discriminator_model(channels=channels,dcgan_dis_n_layers = n_layers, z_dim=z_dim),
                   inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim, x_dim=x_dim)),
        lambda: MDGANv1(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                  models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers),
                  encoder=models.mnist_dcgan_encoder_model(z_dim=z_dim, channels=channels)),
        lambda: MDGANv2(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                  models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers),
                  encoder=models.mnist_dcgan_encoder_model(z_dim=z_dim, channels=channels)),
        lambda: UnrolledGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                        models.mnist_dcgan_discriminator_model(batchnorm=False,n_layers = n_layers,channels=channels)),
        lambda: GDPPGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                    models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers, return_feature_map=True)),
        lambda: WGAN_GP(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                    models.mnist_dcgan_discriminator_model(channels=channels,n_layers = n_layers,batchnorm=False),lam=10.0),
        lambda: PACGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
               models.mnist_dcgan_discriminator_model(channels=channels * 2, n_layers=n_layers), pack_nr=2, name='PACGAN2'),
        lambda: DiverseGAN(models.mnist_dcgan_generator_model(z_dim=z_dim, channels=channels),
                           models.mnist_dcgan_discriminator_model(channels=channels, n_layers=n_layers,
                                                                  return_feature_map=True),
                           diversity_loss_func=losses.frobenius, name="Frobenius2GAN"),
    ]

    cifar_results_i = {}
    print('**** STACKED MNIST ****')
    x, y = data.load_stacked_mnist()
    x = np.rollaxis(x.reshape((-1, 3, 28, 28)), 1, 4)

    for gan_f in gans:
        name = None
        timings = []
        for i in range(n_rep):
            gan = gan_f()
            name = gan.name
            gan.train(x, batch_size=batch_size, steps=5, save_samples_every=-1, log_losses=False,
                      logdir='out_STACKEDMNIST') # init TF graph
            start = timer() # actually time training
            gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=-1, log_losses=False,
                      logdir='out_STACKEDMNIST')
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

