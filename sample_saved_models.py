from lib.gans import *
from lib import models
import os
import matplotlib.pyplot as plt


# requires there to be a logdir/models directory with all saved models in a separate subfolder.
# The subfolder should have the same name as the original saved model

if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')# To not use GPU

    logdir = 'OUT_CIFAR10_test'
    # logdir = 'out_StackedMNIST_64_2'
    type = 'cifar10'  # smnist, cifar10, cifar100, stl10
    paths = [ os.path.join(f.path, f.name) for f in os.scandir(os.path.join(logdir, 'models')) if f.is_dir() ]
    names = [f.name for f in os.scandir(os.path.join(logdir, 'models')) if f.is_dir() ]
    rows = 8
    cols = 8
    grid_size = rows*cols

    out_dir = 'samples'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    channels=3
    z_dim = 100
    sampler = default_z_sampler

    file_type = '.png' # .pdf .png

    for path,name in zip(paths,names):

        if type.lower() == 'smnist':
            generator = models.mnist_dcgan_generator_model(z_dim,channels)
        elif type.lower() == 'cifar10' or type.lower() == 'cifar100':
            generator = models.cifar10_dcgan_generator_model(z_dim)
        elif type.lower() == 'stl10':
            generator = models.stl10_dcgan_generator_model(z_dim)
        else:
            raise AssertionError('Unknown type.')

        generator.load_weights(path)
        noise = sampler([grid_size, z_dim])
        samples = generator.predict(x=noise,batch_size=grid_size)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0, wspace=0)
        for i in range(samples.shape[0]):
            plt.subplot(rows, cols, i + 1)
            plt.imshow((samples[i, :, :, :] + 1) * 0.5, aspect='auto')
            plt.axis('off')
        plt.savefig(os.path.join(out_dir, name + file_type))
        plt.close()


