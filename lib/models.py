import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp


def to_shape(a):
    if type(a) == tuple:
        input_shape = a
    elif type(a) == list:
        input_shape = tuple(a)
    elif type(a) == int:
        input_shape = (a,)
    else:
        raise AssertionError("Shape type not recognized, expected tuple, list or int.")
    return input_shape


def mnist_dcgan_generator_model(z_dim, channels=1, batchnorm=True):
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(z_dim,), activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


def mnist_dcgan_generator_model_original_veegan(z_dim, channels=1, batchnorm=True):
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(2 * 2 * 512, use_bias=True, input_shape=(z_dim,), activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Reshape((2, 2, 512)))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='relu')) #padding='same'
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2),padding='same', output_padding=0, use_bias=True, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(
        layers.Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', use_bias=True, activation='tanh'))
    return model


def mnist_dcgan_discriminator_model(dropout_rate=0.0, output_activation=None, return_feature_map=False, channels=1,
                                    batchnorm=True, n_layers = 4, layer_1_filters = 64):
    input = layers.Input(shape=(28, 28, channels))
    conv = layers.Conv2D(layer_1_filters, (5, 5), strides=(2, 2), padding='same')(input)
    conv = layers.LeakyReLU()(conv)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)
    prev_filter_nr = layer_1_filters
    for i in range(n_layers - 1):
        new_filter_nr = prev_filter_nr * 2
        conv = layers.Conv2D(new_filter_nr, (5, 5), strides=(2, 2), padding='same')(conv)  # 128
        conv = layers.LeakyReLU()(conv)
        if batchnorm:
            conv = layers.BatchNormalization()(conv)
        if dropout_rate > 0:
            conv = layers.Dropout(dropout_rate)(conv)
        prev_filter_nr = new_filter_nr

    conv = layers.Flatten()(conv)
    out = layers.Dense(1, activation=output_activation)(conv)  # logits = True if linear activation i.e. none

    dis_model = tf.keras.Model(inputs=input, outputs=out)
    if return_feature_map:
        feature_map_model = tf.keras.Model(inputs=input, outputs=conv)
        return dis_model, feature_map_model
    else:
        return dis_model


def cifar10_dcgan_generator_model(z_dim, batchnorm=True):
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(z_dim,), activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def stl10_dcgan_generator_model(z_dim, batchnorm=True):
    model = tf.keras.Sequential()
    model.add(layers.Dense(12 * 12 * 256, use_bias=False, input_shape=(z_dim,), activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Reshape((12, 12, 256)))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model



def cifar10_dcgan_discriminator_model(dropout_rate=0.0, batchnorm=True, output_activation=None,
                                      return_feature_map=False, layer_1_filters = 64, n_layers = 3, image_input_shape=(32, 32, 3)):
    input = layers.Input(shape=image_input_shape)
    conv = layers.Conv2D(layer_1_filters, (5, 5), strides=(2, 2), padding='same')(input)
    conv = layers.LeakyReLU()(conv)
    if dropout_rate > 0:
        conv = layers.Dropout(dropout_rate)(conv)
    prev_filter_nr = layer_1_filters
    for i in range(n_layers - 1):
        new_filter_nr = prev_filter_nr * 2
        conv = layers.Conv2D(new_filter_nr, (5, 5), strides=(2, 2), padding='same')(conv)  # 128
        conv = layers.LeakyReLU()(conv)
        if batchnorm:
            conv = layers.BatchNormalization()(conv)
        if dropout_rate > 0:
            conv = layers.Dropout(dropout_rate)(conv)
        prev_filter_nr = new_filter_nr

    conv = layers.Flatten()(conv)
    out = layers.Dense(1, activation=output_activation)(conv)  # logits = True if linear activation i.e. none

    dis_model = tf.keras.Model(inputs=input, outputs=out, name='discriminator')
    if return_feature_map:
        feature_map_model = tf.keras.Model(inputs=input, outputs=conv)
        return dis_model, feature_map_model
    else:
        return dis_model


# VEEGAN requires a special discriminator model, noise input is concatenated with the feature map before classification
def cifar10_VEEGAN_discriminator_model(z_dim, dropout_rate=0.0, batchnorm=True, output_activation=None,
                                       return_feature_map=False,
                                       n_hidden=512, n_layers=2, activation='relu', dcgan_dis_n_layers = 3,image_input_shape=(32, 32, 3)):
    standard_model, feature_map_model = cifar10_dcgan_discriminator_model(dropout_rate=dropout_rate,
                                                                          batchnorm=batchnorm,
                                                                          output_activation=output_activation,
                                                                          return_feature_map=True, n_layers = dcgan_dis_n_layers,
                                                                          image_input_shape = image_input_shape)
    noise_input = layers.Input(shape=(z_dim,))
    cifar10_input = layers.Input(shape=image_input_shape)
    f_out = feature_map_model(cifar10_input)
    concatenated = layers.concatenate([noise_input, f_out])
    dense = layers.Dense(n_hidden, activation=activation)(concatenated)
    for _ in range(n_layers - 1):
        dense = layers.Dense(n_hidden, activation=activation)(dense)

    out = layers.Dense(1, activation=output_activation)(dense)  # logits = True if linear activation i.e. none
    dis_model = tf.keras.Model(inputs=[noise_input, cifar10_input], outputs=out)
    if return_feature_map:
        return dis_model, feature_map_model
    else:
        return dis_model


# VEEGAN requires a special discriminator model, noise input is concatenated with the feature map before classification
def mnist_VEEGAN_discriminator_model(z_dim, dropout_rate=0.0, output_activation=None, return_feature_map=False,
                                     channels=1,
                                     n_hidden=512, n_layers=2, activation='relu', dcgan_dis_n_layers = 2):
    standard_model, feature_map_model = mnist_dcgan_discriminator_model(dropout_rate=dropout_rate,
                                                                        output_activation=output_activation,
                                                                        return_feature_map=True, channels=channels, n_layers = dcgan_dis_n_layers)
    noise_input = layers.Input(shape=(z_dim,))
    mnist_input = layers.Input(shape=(28, 28, channels))
    f_out = feature_map_model(mnist_input)
    concatenated = layers.concatenate([noise_input, f_out])
    dense = layers.Dense(n_hidden, activation=activation)(concatenated)
    for _ in range(n_layers - 1):
        dense = layers.Dense(n_hidden, activation=activation)(dense)

    out = layers.Dense(1, activation=output_activation)(dense)  # logits = True if linear activation i.e. none
    dis_model = tf.keras.Model(inputs=[noise_input, mnist_input], outputs=out)
    if return_feature_map:
        return dis_model, feature_map_model
    else:
        return dis_model


def fully_connected_generator_model(z_dim, x_dim, n_hidden=128, n_layers=2, activation='tanh'):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(z_dim,)))
    for _ in range(n_layers):
        model.add(layers.Dense(n_hidden, activation=activation))
    model.add(layers.Dense(x_dim))
    return model


def fully_connected_discriminator_model(x_dim, n_hidden=128, n_layers=2, activation='tanh', output_activation=None,
                                        return_feature_map=False):
    inp = layers.Input(shape=(x_dim,))
    dense = layers.Dense(n_hidden, activation)(inp)
    for _ in range(n_layers - 1):
        dense = layers.Dense(n_hidden, activation=activation)(dense)
    out = layers.Dense(1, activation=output_activation)(dense)
    dis_model = tf.keras.Model(inputs=inp, outputs=out)
    if return_feature_map:
        feature_map_model = tf.keras.Model(inputs=inp, outputs=dense)
        return dis_model, feature_map_model
    else:
        return dis_model


# encodes x_dim to z_dim, as used in e.g. MDGANv2
def fully_connected_encoder_model(z_dim, x_dim, activation='tanh', n_hidden=128, n_layers=1):
    input = layers.Input(shape=(x_dim,))
    dense = layers.Dense(n_hidden, activation=activation)(input)
    for _ in range(n_layers - 1):
        dense = layers.Dense(n_hidden, activation=activation)(dense)
    out = layers.Dense(z_dim)(dense)
    enc_model = tf.keras.Model(inputs=input, outputs=out)
    return enc_model


# Maps x_dim to z_dim with a stochastic output. Assumes normal distribution with scale = 1. As used in e.g. VEEGAN
def fully_connected_stochastic_inverse_generator(x_dim, z_dim, activation='relu', n_layers=2, n_hidden=128):
    model = tf.keras.Sequential()
    x_dim_shape = to_shape(x_dim)
    model.add(layers.Input(shape=x_dim_shape))
    if len(x_dim_shape) > 1:
        model.add(layers.Flatten())
    for _ in range(n_layers):
        model.add(layers.Dense(n_hidden, activation=activation))
    model.add(layers.Dense(z_dim))
    model.add(tfp.layers.DistributionLambda(  # replaces stochastic_tensor
        make_distribution_fn=
        lambda t: tfp.distributions.Normal(
            loc=t, scale=1 * tf.ones((z_dim,))))  # ,convert_to_tensor_fn=lambda s: s.sample(batch_size)
    )
    return model


# As used in MDGANv2 and described in appendix B, reversed generator
def mnist_dcgan_encoder_model(z_dim, batchnorm=True, channels=1):
    model = tf.keras.Sequential(name='encoder')
    model.add(layers.Input(shape=(28, 28, channels)))
    model.add(layers.Conv2D(channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(z_dim, use_bias=False))

    return model


def cifar10_dcgan_encoder_model(z_dim, batchnorm=True, image_input_shape=(32, 32, 3)):
    model = tf.keras.Sequential(name='encoder')
    model.add(layers.Input(shape=image_input_shape))
    model.add(layers.Conv2D(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(z_dim))
    return model

def stl10_dcgan_encoder_model(z_dim, batchnorm=True, image_input_shape=(96, 96, 3)):
    model = tf.keras.Sequential(name='encoder')
    model.add(layers.Input(shape=image_input_shape))
    model.add(layers.Conv2D(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='relu'))
    if batchnorm:
        model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(z_dim))
    return model

# Simple MNIST CNN used in calculation of the KL-divergence
def mnist_cnn(dropout_rate=0.5):
    model = tf.keras.Sequential(name='cnn_model')
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(10, activation='softmax'))
    return model
