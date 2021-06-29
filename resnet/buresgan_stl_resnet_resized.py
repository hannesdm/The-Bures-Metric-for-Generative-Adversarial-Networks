"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.layernorm
import tflib.save_images
import tflib.stl10
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')



##CUSTOM
cross_entropy_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
##CUSTOM END

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'data'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

LOW_MEM = True # reduces memory usage during IS calculation but slower
LOG_FILE='buresgan_stl_resnet_out.txt'


BATCH_SIZE = 64 # Critic batch size
BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 300000 # How many iterations to train for
DIM_G = 200 # Generator dimensionality
DIM_D = 200 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = True # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 48*48*3 # Number of pixels in CIFAR10 (32*32*3)
LR = 5e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 1 # Critic steps per generator steps
INCEPTION_FREQUENCY = 10000 # How frequently to calculate Inception score

CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

NMBDIS = 4
NMBGEN = 4
LOG_DIR = 'STL_1_best_big_resized_v2' + '/'
Path(os.path.dirname(LOG_DIR)).mkdir(parents=True, exist_ok=True)

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.nn.avg_pool_v2(output, strides=(2,2), ksize=(2,2), padding='SAME', data_format='NCHW')
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.nn.avg_pool_v2(output, strides=(2, 2), ksize=(2,2), padding='SAME', data_format='NCHW')
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    # output = tf.concat([output, output, output, output], axis=1)
    # output = tf.transpose(output, [0,2,3,1])
    # output = tf.depth_to_space(output, 2)
    # output = tf.transpose(output, [0,3,1,2])
    output = tf.keras.backend.resize_images(output,2,2,'channels_first', interpolation='nearest')
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)    
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)            
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output

def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)    
    output = nonlinearity(output)            
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output

def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([int(n_samples), 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 6*6*DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 6, 6])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    #output = ResidualBlock('Generator.4', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 48, 48])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample='down', labels=labels) 
    output = ResidualBlock('Discriminator.5', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.6', DIM_D, DIM_D, 3, output, resample=None, labels=labels) 
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    logits = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    logits = tf.reshape(logits, [-1])
    return logits, output

# Matrix square root using the Newton-Schulz method
def sqrt_newton_schulz(A, iterations=20, dtype='float64'):
    dim = int(A.shape[0])
    normA = tf.norm(A)
    Y = tf.divide(A, normA)
    I = tf.eye(dim, dtype=dtype)
    Z = tf.eye(dim, dtype=dtype)
    for i in range(iterations):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    sqrtA = Y * tf.sqrt(normA)
    return sqrtA

def wasserstein_bures_covariance(fake_phi, real_phi, epsilon=10e-14, sqrtm_func=sqrt_newton_schulz, normalize=True,
                                 dtype='float64', weight=1.):
    if dtype == 'float64':
        fake_phi = tf.cast(fake_phi, tf.float64)
        real_phi = tf.cast(real_phi, tf.float64)

    batch_size = fake_phi.shape[0]
    h_dim = int(fake_phi.shape[1])

    # Center and normalize
    fake_phi = fake_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(fake_phi, axis=0,
                                                                                            keepdims=True)
    real_phi = real_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(real_phi, axis=0,
                                                                                            keepdims=True)
    if normalize:
        fake_phi = tf.nn.l2_normalize(fake_phi, 1)
        real_phi = tf.nn.l2_normalize(real_phi, 1)

    # bures
    C1 = tf.transpose(fake_phi) @ fake_phi
    C1 = C1 + epsilon * tf.eye(h_dim, dtype=dtype)
    C2 = tf.transpose(real_phi) @ real_phi
    C2 = C2 + epsilon * tf.eye(h_dim, dtype=dtype)

    bures = tf.linalg.trace(C1) + tf.linalg.trace(C2) - 2 * tf.linalg.trace(sqrtm_func(C1 @ C2))

    return weight * bures

def cross_entropy_discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy_from_logits(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss

with tf.Session() as session:

    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    fake_data_splits = []
    for i, device in enumerate(DEVICES):
        with tf.device(device):
            fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

    all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
    all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

    DEVICES_B = DEVICES[:int(len(DEVICES)/2)]
    DEVICES_A = DEVICES[int(len(DEVICES)/2):]

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    for i, device in enumerate(DEVICES_A):
        with tf.device(device):
            real_and_fake_data = tf.concat([
                all_real_data_splits[i], 
                all_real_data_splits[len(DEVICES_A)+i], 
                fake_data_splits[i], 
                fake_data_splits[len(DEVICES_A)+i]
            ], axis=0)
            real_and_fake_labels = tf.concat([
                labels_splits[i], 
                labels_splits[len(DEVICES_A)+i],
                labels_splits[i],
                labels_splits[len(DEVICES_A)+i]
            ], axis=0)
            disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)
            disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
            disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]
            disc_costs.append(cross_entropy_discriminator_loss(disc_real, disc_fake))

    disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)

    disc_acgan = tf.constant(0.)
    disc_acgan_acc = tf.constant(0.)
    disc_acgan_fake_acc = tf.constant(0.)
    disc_cost = disc_wgan

    disc_params = lib.params_with_name('Discriminator.')

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.


    gen_costs = []
    gen_acgan_costs = []
    for device in DEVICES:
        with tf.device(device):
            n_samples = int(GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES))
            fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
            G_out = Generator(n_samples, fake_labels)
            fake_logits, phi_fake = Discriminator(G_out, fake_labels)
            gen_cross_entropy = cross_entropy_from_logits(tf.ones_like(fake_logits), fake_logits)
            real_batch = tf.concat([all_real_data_splits[0], all_real_data_splits[1]], axis=0)
            real_logits, phi_real = Discriminator(real_batch, fake_labels)
            diversity_loss = wasserstein_bures_covariance(phi_fake, phi_real)
            gen_costs.append(0.5 * (gen_cross_entropy + tf.cast(diversity_loss, 'float32')))

    gen_cost = (tf.add_n(gen_costs) / len(DEVICES))



    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # Function for generating samples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 48, 48)), LOG_DIR + 'samples_{}.png'.format(frame))

    # Function for calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(100, fake_labels_100)


    def get_inception_score(n):
        train_images, train_labels, test_images, test_labels = lib.stl10.load_stl10_images(resize_shape=(48,48))
        train_images = train_images.reshape((-1, 3, 48, 48)).transpose(0, 2, 3, 1)
        if LOW_MEM:
            with tempfile.NamedTemporaryFile() as ntf:
                arr = np.memmap(ntf, dtype='float32', mode='w+', shape=(50000, 48, 48, 3))
                for i in range(int(n / 100)):
                    ims = session.run(samples_100)
                    ims = ((ims + 1.) * (255 / 2)).astype('int32')
                    ims = ims.reshape((-1, 3, 48, 48)).transpose(0, 2, 3, 1)
                    arr[i * 100:(i + 1) * 100] = ims
                inception_score, is_std = lib.inception_score.get_inception_score(arr)
                fid = lib.inception_score.calculate_FID_from_images(real_images=train_images, fake_images=arr)
        else:
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples + 1.) * (255 / 2)).astype('int32')
            all_samples = all_samples.reshape((-1, 3, 48, 48)).transpose(0, 2, 3, 1)
            inception_score, is_std = lib.inception_score.get_inception_score(list(all_samples))
            fid = lib.inception_score.calculate_FID_from_images(real_images=train_images, fake_images=all_samples)
        return inception_score, is_std, fid

    train_gen, dev_gen = lib.stl10.load(BATCH_SIZE,resize_shape=(48,48))
    def inf_train_gen():
        while True:
            for images,_labels in train_gen():
                yield images,_labels


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print("{} Params:".format(name))
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g is None:
                print("\t{} ({}) [no grad!]".format(v.name, shape_str))
            else:
                print("\t{} ({})".format(v.name, shape_str))
        print("Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        ))

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    best_is = 0.0
    best_fid = 1000.
    for iteration in range(ITERS):
        start_time = time.time()

        _data, _labels = next(gen)
        if iteration > 0:
            _ = session.run([gen_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels,_iteration:iteration})

        _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

        lib.plot.plot('cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY - 1:
            inception_score, is_std, fid = get_inception_score(50000)
            if inception_score > best_is:
                best_is = inception_score
            if fid < best_fid:
                best_fid = fid
            lib.plot.plot('Best inception_50k', best_is)
            lib.plot.plot('Best FID_50k', best_fid)
            lib.plot.plot('inception_50k', inception_score)
            lib.plot.plot('inception_50k_std', is_std)
            lib.plot.plot('FID_50k', fid)

        # Calculate dev loss and generate samples every 5000 iters
        if (iteration + 1) % 5000 == 0:
            dev_disc_costs = []
            for images,_labels in dev_gen():
                _dev_disc_cost = session.run([disc_cost], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush(LOG_DIR,LOG_FILE)

        lib.plot.tick()
