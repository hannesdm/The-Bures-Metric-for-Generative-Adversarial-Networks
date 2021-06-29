import gc

from lib import evaluate, models, data, linalg
from lib.evaluate import InceptionScorer
from lib.gans import *
from tqdm import trange
import numpy as np
import sys
from timeit import default_timer as timer


def main():
    # tf.config.set_visible_devices([], 'GPU') #To not use GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    name = sys.argv[1]
    GANname = name
    name = 'STL_final_' + name 

    
    if os.path.exists(name):
        os.remove(name)
    
    def write_to_console_file(txt):
        with open(name,'a') as f:
            f.write(txt+'\n')
            print(txt)


    z_dim = 100
    channels = 3
    x_dim = (96, 96, channels)
    batch_size = 256
    inception_batch_size = 1  # batch size used for the inception scores
    sample_nr = 5000  # How many samples to use and evaluate
    steps = 150000
    n_rep = 5

    REGULARIZOR_sqrt = 1e-14
    WEIGHT_Bures = 1
    ITERATION_sqrt = 15
    NORMALIZATION = True

    print('--------------------------------------------------------')
    write_to_console_file('\n Regularizor sqrt= {} '.format(REGULARIZOR_sqrt))
    write_to_console_file('\n Iterations sqrt= {} '.format(ITERATION_sqrt))
    write_to_console_file('\n Weight in front of Bures= {} '.format(WEIGHT_Bures))
    write_to_console_file('\n Normalization= {} '.format(NORMALIZATION))

    print('--------------------------------------------------------')

    sqrtm = lambda A: linalg.sqrt_newton_schulz(A, iterations=ITERATION_sqrt)

    div_func = lambda h1, h2: losses.wasserstein_bures_kernel(h1, h2, epsilon=REGULARIZOR_sqrt, sqrtm_func=sqrtm,
                                                              normalize=NORMALIZATION, dtype='float64',
                                                              weight=WEIGHT_Bures)

    image_input_shape = (96,96,3)
    x_sampler_args = {'sample_cache_nr': 'batch'} # only sample 1 batch at a time to limit memory usage
    if GANname == 'AlternatingBures':
        def create_gan_model():
            return AlternatingBuresGAN(models.stl10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True, n_layers=4,image_input_shape=image_input_shape),
                                        diversity_loss_func=div_func, x_sampler_args=x_sampler_args)
    elif GANname == 'Bures':
        def create_gan_model():
            return BuresGAN(models.stl10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True, n_layers=4, image_input_shape=image_input_shape)
                             , diversity_loss_func=div_func, x_sampler_args=x_sampler_args)
    elif GANname == 'GAN':
        def create_gan_model():
            return GAN(models.stl10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(n_layers=4, image_input_shape=image_input_shape),
                        x_sampler_args=x_sampler_args)
    elif GANname == 'VEEGAN':
        def create_gan_model():
            return VEEGAN(models.stl10_dcgan_generator_model(z_dim=z_dim), models.cifar10_VEEGAN_discriminator_model(dcgan_dis_n_layers=4, image_input_shape=image_input_shape,z_dim=z_dim),
                           inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim, x_dim=x_dim), x_sampler_args=x_sampler_args)
    elif GANname == 'MDGANv1':
        def create_gan_model():
            return MDGANv1(models.stl10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(n_layers=4, image_input_shape=image_input_shape),
                          encoder=models.stl10_dcgan_encoder_model(z_dim=z_dim, image_input_shape=image_input_shape),x_sampler_args=x_sampler_args)
    elif GANname == 'MDGANv2':
        def create_gan_model():
            return MDGANv2(models.stl10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(n_layers=4, image_input_shape=image_input_shape),
                          encoder=models.stl10_dcgan_encoder_model(z_dim=z_dim, image_input_shape=image_input_shape),x_sampler_args=x_sampler_args)
    elif GANname == 'UnrolledGAN':
        def create_gan_model():
            return UnrolledGAN(models.stl10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(batchnorm=False, n_layers=4,image_input_shape=image_input_shape),
                                x_sampler_args=x_sampler_args)  # UnrolledGAN collapses with bathnorm = True
    elif GANname == 'GDPPGAN':
        def create_gan_model():
            return GDPPGAN(models.stl10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(return_feature_map=True, n_layers=4, image_input_shape=image_input_shape),
                            x_sampler_args=x_sampler_args)
    elif GANname == 'WGANGP':
        def create_gan_model():
            return WGAN_GP(models.stl10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(n_layers=4, image_input_shape=image_input_shape, batchnorm=False), lam=10.0,
                            x_sampler_args=x_sampler_args) # WGAN should not have batchnorm in discriminator
    elif GANname == 'PACGAN':
        def create_gan_model():
            return PACGAN(models.stl10_dcgan_generator_model(z_dim=z_dim),
                           models.cifar10_dcgan_discriminator_model(n_layers=4,image_input_shape=(96,96,3 * 2)), name='PACGAN2', pack_nr=2)
    elif GANname == 'frobenius':
        def create_gan_model():
            return DiverseGAN(models.stl10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True, n_layers=4, image_input_shape=image_input_shape)
                             , diversity_loss_func=losses.frobenius, x_sampler_args=x_sampler_args)
    else:
        raise ValueError('GAN name is unknown '+ GANname)

    print('**** STL 10 ****')
    cifar_results_i = {}
    inception_scorer = evaluate.InceptionScorer()
    print("Loading data...")
    x = data.load_stl10_images()
    for i in range(n_rep):
      write_to_console_file("Iteration: {}/{}".format(i + 1, n_rep))
      gan = create_gan_model()
      write_to_console_file('---- ' + GANname + ' ----')
      start = timer()
      gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=25000, log_losses=False,logdir='out_STL_final')
      end = timer()
      time = (end - start)
      write_to_console_file('Evaluating ' + GANname + '...')
      ivo = evaluate.get_inference_via_optimization(gan, x, sample_nr=sample_nr, batch_size=batch_size)
      write_to_console_file("IvO: {}".format(ivo))
      samples = gan.sample_generator(nr=sample_nr,batch_size=batch_size)
      inception_score = inception_scorer.inception_score(samples, batch_size=inception_batch_size)
      write_to_console_file('Inception Score: {}'.format(inception_score))
      fid = inception_scorer.frechet_inception_distance(real_images=x,fake_images=samples, batch_size=inception_batch_size, cache_real_output=True)
      write_to_console_file('Frechet Inception Distance: {}'.format(fid))
      sliced_wasserstein = evaluate.sliced_wasserstein_distance(samples, x, sample_nr=sample_nr)
      sliced_wasserstein = tf.reduce_mean(sliced_wasserstein).numpy()
      write_to_console_file('Sliced Wasserstein Distance: {}'.format(sliced_wasserstein))
      if GANname not in cifar_results_i:
        cifar_results_i[GANname] = []
      cifar_results_i[GANname].append([ivo,inception_score, fid, sliced_wasserstein,time])

      gan.save_generator('out_STL', name= GANname + '_' + str(i))
      tf.keras.backend.clear_session()  # free memory
      del gan
  
    # Print results 
    for k,v in cifar_results_i.items():       
      write_to_console_file('\n FINAL RESULTS STL' + k)
      arr = np.asarray(v)
      means = np.mean(arr, axis=0)
      stds = np.std(arr, axis=0)
      means = np.round(means,decimals = 4)
      stds = np.round(stds,decimals = 4)
        
      write_to_console_file('IvO: {}({})'.format(means[0], stds[0]))
      write_to_console_file('Inception Score: {}({})'.format(means[1], stds[1]))
      write_to_console_file('Frechet Inception Distance: {}({})'.format(means[2], stds[2]))
      write_to_console_file('Sliced Wasserstein Distance: {}({})'.format(means[3], stds[3]))
      write_to_console_file('Time: {}({})'.format(means[4], stds[4]))



if __name__ == '__main__':

    main()

