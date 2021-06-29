from lib.gans import *
from lib import evaluate, models, data,linalg, losses
from timeit import default_timer as timer
import numpy as np

if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU')

    if os.path.exists('results_file_Bures_primal'):
        os.remove('results_file_Bures_primal')


    def write_to_console_file(txt):
        with open('results_file_Bures_primal', 'a') as f:
            f.write(txt + '\n')
            print(txt)


    def create_gan_models():
        return [
            MDGANv1(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim), models.fully_connected_discriminator_model(x_dim=x_dim), encoder=models.fully_connected_encoder_model(z_dim=z_dim, x_dim=x_dim)),
            MDGANv2(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim), models.fully_connected_discriminator_model(x_dim=x_dim), encoder=models.fully_connected_encoder_model(z_dim=z_dim, x_dim=x_dim)),
            GAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                 models.fully_connected_discriminator_model(x_dim=x_dim)),
            WGAN_GP(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                     models.fully_connected_discriminator_model(x_dim=x_dim), lam=5.0, n_critic=10),
            VEEGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                    models.fully_connected_discriminator_model(x_dim=z_dim + x_dim),
                    inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim, x_dim=x_dim)),
            
            UnrolledGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                         models.fully_connected_discriminator_model(x_dim=x_dim)),
            GDPPGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                     models.fully_connected_discriminator_model(x_dim=x_dim, return_feature_map=True)),
            BuresGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                     models.fully_connected_discriminator_model(x_dim=x_dim, return_feature_map=True)),
            AlternatingBuresGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                     models.fully_connected_discriminator_model(x_dim=x_dim, return_feature_map=True)),
            PACGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                   models.fully_connected_discriminator_model(x_dim=x_dim * 2, return_feature_map=True),
                   pack_nr=2, name='PACGAN2'),
            DiverseGAN(models.fully_connected_generator_model(z_dim=z_dim, x_dim=x_dim),
                       models.fully_connected_discriminator_model(x_dim=x_dim, return_feature_map=True),
                       diversity_loss_func=losses.frobenius, name='Frobenius2GAN'),
        ]

    z_dim = 256
    x_dim = 2
    x_samples = 10000
    eval_samples = 2500
    batch_size = 256
    steps = 25000
    n_rep = 10

    grid_results = {}
    ring_results = {}

    ### GRID ###
    for i in range(n_rep):
        gans = create_gan_models()
        x = data.sample_ring(x_samples)

        for gan in gans:
            write_to_console_file('---- ' + gan.name + ' ----')
            start = timer()
            gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=1000, log_losses=False,
                      logdir='out_ring')
            end = timer()
            time = (end - start)
            samples = gan.sample_generator(eval_samples)
            nr_modes_captured, percentage_within_3std = evaluate.evaluate_ring(samples)
            write_to_console_file('Number of Modes Captured: {}'.format(nr_modes_captured))
            write_to_console_file('Number of Points Falling Within 3 std. of the Nearest Mode {}/{} = {}'
                                  .format(percentage_within_3std * eval_samples, eval_samples, percentage_within_3std))

            if gan.name not in ring_results:
                ring_results[gan.name] = []
            ring_results[gan.name].append([nr_modes_captured, percentage_within_3std, time])

            tf.keras.backend.clear_session()  # free memory
            del gan

        del gans

    for k, v in ring_results.items():
        write_to_console_file('\n FINAL RESULTS RING' + k)
        arr = np.asarray(v)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        write_to_console_file('Number of Modes Captured: {} +- {}'.format(means[0], stds[0]))
        write_to_console_file('Number of Points Falling Within 3 std. of the Nearest Mode {}/{} = {} +- {}'
                              .format(means[1] * eval_samples, eval_samples, means[1], stds[1]))

        write_to_console_file('training time {} +- {}'
                              .format(means[2], stds[2]))

    ### GRID ###

    write_to_console_file('\n **** GRID ****')

    for i in range(n_rep):
        gans = create_gan_models()
        x = data.sample_grid(x_samples)

        for gan in gans:
            write_to_console_file('---- ' + gan.name + ' ----')
            start = timer()
            gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=1000, log_losses=False,
                      logdir='out_grid')
            end = timer()
            time = (end - start)
            samples = gan.sample_generator(eval_samples)
            nr_modes_captured, percentage_within_3std = evaluate.evaluate_grid(samples)
            write_to_console_file('Number of Modes Captured: {}'.format(nr_modes_captured))
            write_to_console_file('Number of Points Falling Within 3 std. of the Nearest Mode {}/{} = {}'
                                  .format(percentage_within_3std * eval_samples, eval_samples, percentage_within_3std))

            if gan.name not in grid_results:
                grid_results[gan.name] = []
            grid_results[gan.name].append([nr_modes_captured, percentage_within_3std, time])

            tf.keras.backend.clear_session()  # free memory
            del gan

        del gans

    for k, v in grid_results.items():
        write_to_console_file('\n FINAL RESULTS GRID' + k)
        arr = np.asarray(v)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        write_to_console_file('Number of Modes Captured: {} +- {}'.format(means[0], stds[0]))
        write_to_console_file('Number of Points Falling Within 3 std. of the Nearest Mode {}/{} = {} +- {}'
                              .format(means[1] * eval_samples, eval_samples, means[1], stds[1]))

        write_to_console_file('training time step1 {} +- {}'
                              .format(means[2], stds[2]))
