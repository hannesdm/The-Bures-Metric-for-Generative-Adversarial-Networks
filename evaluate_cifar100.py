from lib.gans import *
from lib import evaluate, models, data, linalg
from lib.evaluate import InceptionScorer
from lib.gans import *
from tqdm import trange
from timeit import default_timer as timer
import numpy as np
import sys


if __name__ == '__main__':

    # tf.config.set_visible_devices([], 'GPU') #To not use GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    name = sys.argv[1]
    GANname = name
    name = 'CIFAR100_final_' + name 

    
    if os.path.exists(name):
        os.remove(name)
    
    def write_to_console_file(txt):
        with open(name,'a') as f:
            f.write(txt+'\n')
            print(txt)

    z_dim = 100
    channels = 3
    x_dim = (32, 32, channels)
    batch_size = 256
    inception_batch_size = 1 # batch size used for the inception scores
    sample_nr = 10000 # How many samples to use and evaluate TODO How many are sampled in previous work?
    steps = 100000
    n_rep = 10

    REGULARIZOR_sqrt=1e-14
    WEIGHT_Bures=1
    ITERATION_sqrt = 15
    NORMALIZATION = True
    DUAL = True
    x_sampler= 'npuniform' # 'uniform, shuffle, npuniform'
    x_sampler_args={'sample_cache_nr':'all'} # precalculate an amount of batches, all means cache the entire dataset, should be increments of batch_size e.g. 5*batch_size to cache 5 batches
    
    print('--------------------------------------------------------')
    write_to_console_file('\n Regularizor sqrt= {} '.format(REGULARIZOR_sqrt))
    write_to_console_file('\n Iterations sqrt= {} '.format(ITERATION_sqrt))    
    write_to_console_file('\n Weight in front of Bures= {} '.format(WEIGHT_Bures))
    write_to_console_file('\n Normalization= {} '.format(NORMALIZATION))
    write_to_console_file('\n Repetitions = {} '.format(n_rep))
    write_to_console_file('\n Steps = {} '.format(steps))
    print('--------------------------------------------------------')

    sqrtm = lambda A: linalg.sqrt_newton_schulz(A, iterations=ITERATION_sqrt)

    div_func = lambda h1, h2: losses.wasserstein_bures_kernel(h1, h2, epsilon=REGULARIZOR_sqrt,sqrtm_func=sqrtm, normalize=NORMALIZATION, dtype='float64',weight=WEIGHT_Bures)

    if GANname == 'AlternatingBures':
        def create_gan_model():
            return AlternatingBuresGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True), diversity_loss_func=div_func)  
    elif GANname == 'Bures':
        def create_gan_model():
            return BuresGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True), diversity_loss_func=div_func)
    elif GANname == 'GAN':
        def create_gan_model():
            return GAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model())
    elif GANname == 'VEEGAN':
        def create_gan_model():
            return VEEGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_VEEGAN_discriminator_model(z_dim=z_dim),
                   inverse_generator=models.fully_connected_stochastic_inverse_generator(z_dim=z_dim, x_dim=x_dim))
    elif GANname == 'MDGANv1':
        def create_gan_model():
            return MDGANv1(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(), encoder=models.cifar10_dcgan_encoder_model(z_dim=z_dim))
    elif GANname == 'MDGANv2':
        def create_gan_model():
            return MDGANv2(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(), encoder=models.cifar10_dcgan_encoder_model(z_dim=z_dim))
    elif GANname == 'UnrolledGAN':
        def create_gan_model():
            return UnrolledGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(batchnorm=False))
    elif GANname == 'GDPPGAN':
        def create_gan_model():
            return GDPPGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(return_feature_map=True))
    elif GANname == 'WGANGP':
        def create_gan_model():
            return WGAN_GP(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(batchnorm=False), lam=10.0)
    elif GANname == 'PACGAN':
        def create_gan_model():
            return PACGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim), models.cifar10_dcgan_discriminator_model(image_input_shape=(32, 32, 3 * 2)), name='PACGAN2', pack_nr=2)
    elif GANname == 'frobenius':
        def create_gan_model():
            return DiverseGAN(models.cifar10_dcgan_generator_model(z_dim=z_dim),models.cifar10_dcgan_discriminator_model(return_feature_map=True), diversity_loss_func=losses
                              .frobenius)
    else:
        raise ValueError('GAN name is unknown '+ GANname)

    cifar_results_i = {}
    print('**** CIFAR 100 ****')
    x, y = data.load_cifar100_images()
    for i in range(n_rep):
      write_to_console_file("Iteration: {}/{}".format(i + 1, n_rep))
      gan = create_gan_model()
      inception_scorer = evaluate.InceptionScorer()
      write_to_console_file('---- ' + GANname + ' ----')
      start = timer()
      gan.train(x, batch_size=batch_size, steps=steps, save_samples_every=25000, log_losses=False,logdir='out_CIFAR100_final')
      end = timer()
      time = (end - start)
      write_to_console_file('Evaluating ' + GANname + '...')
      ivo = evaluate.get_inference_via_optimization(gan, x, sample_nr=sample_nr, batch_size=batch_size)
      write_to_console_file("IvO: {}".format(ivo))
      samples = gan.sample_generator(nr=sample_nr,batch_size=batch_size)
      inception_score = inception_scorer.inception_score(samples, batch_size=inception_batch_size)
      write_to_console_file('Inception Score: {}'.format(inception_score))
      fid = inception_scorer.frechet_inception_distance(real_images=x,fake_images=samples, batch_size=inception_batch_size)
      write_to_console_file('Frechet Inception Distance: {}'.format(fid))
      sliced_wasserstein = evaluate.sliced_wasserstein_distance(samples, x, sample_nr=sample_nr)
      sliced_wasserstein = tf.reduce_mean(sliced_wasserstein).numpy()
      write_to_console_file('Sliced Wasserstein Distance: {}'.format(sliced_wasserstein))
      if GANname not in cifar_results_i:
        cifar_results_i[GANname] = []
      cifar_results_i[GANname].append([ivo,inception_score, fid, sliced_wasserstein,time])

      gan.save_generator('out_CIFAR100', name= GANname + '_' + str(i))
      tf.keras.backend.clear_session()  # free memory
      del gan
  
    # Print results 
    for k,v in cifar_results_i.items():       
      write_to_console_file('\n FINAL RESULTS CIFAR100' + k)
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


