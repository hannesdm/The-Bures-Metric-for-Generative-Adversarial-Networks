import tensorflow as tf
from tensorflow.python.framework import dtypes
from .linalg import sqrt_newton_schulz

cross_entropy_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def cross_entropy_discriminator_loss(real_output, fake_output, logits=True):
    if logits:
        real_loss = cross_entropy_from_logits(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss + fake_loss) / 2
    return total_loss


def cross_entropy_generator_loss(fake_output, logits=True):
    if logits:
        loss = cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)
    else:
        loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss

def matrix_sqrt_eigen(mat):
    eig_val, eig_vec = tf.linalg.eigh(mat)
    diagonal = tf.linalg.diag(tf.pow(eig_val,0.5))
    mat_sqrt = tf.matmul(diagonal,tf.transpose(eig_vec))
    mat_sqrt = tf.matmul(eig_vec,mat_sqrt)
    return mat_sqrt


def frobenius(fake_phi, real_phi, epsilon=10e-14, normalize=True,
                                     dtype='float64', weight=1.):
    if dtype == 'float64':
            fake_phi = tf.cast(fake_phi, tf.float64)
            real_phi = tf.cast(real_phi, tf.float64)

    batch_size = int(fake_phi.shape[0])
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
    C1 = tf.transpose(fake_phi) @ fake_phi + epsilon * tf.eye(h_dim, dtype=dtype)
    C2 = tf.transpose(real_phi) @ real_phi + epsilon * tf.eye(h_dim, dtype=dtype)

    frob = tf.linalg.norm(C1 - C2)
    frob = frob * frob

    return weight * frob


def wasserstein_bures_kernel(fake_phi, real_phi, sqrtm_func=sqrt_newton_schulz, epsilon=10e-14, normalize=True , dtype='float64',weight=1.,method='NewtonSchultz'):
    if dtype == 'float64':
        fake_phi = tf.cast(fake_phi, dtypes.float64)
        real_phi = tf.cast(real_phi, dtypes.float64)

    batch_size = fake_phi.shape[0]

    # Center and normalize
    fake_phi = fake_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(fake_phi, axis=0, keepdims=True)
    real_phi = real_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(real_phi, axis=0, keepdims=True)
    if normalize:
        fake_phi = tf.nn.l2_normalize(fake_phi, 1)
        real_phi = tf.nn.l2_normalize(real_phi,1)

    K11 = fake_phi @ tf.transpose(fake_phi)
    K11 = K11 + epsilon * tf.eye(batch_size, dtype=dtype)
    K22 = real_phi @ tf.transpose(real_phi)
    K22 = K22 + epsilon * tf.eye(batch_size, dtype=dtype)

    K12 = fake_phi @ tf.transpose(real_phi) + epsilon * tf.eye(batch_size, dtype=dtype)

    if method == 'NewtonSchultz':
        bures = tf.linalg.trace(K11) + tf.linalg.trace(K22) - 2 * tf.linalg.trace(sqrtm_func(K12 @ tf.transpose(K12)))
    else:
        bures = tf.linalg.trace(K11) + tf.linalg.trace(K22) - 2 * tf.linalg.trace(matrix_sqrt_eigen(K12 @ tf.transpose(K12)))

    return weight * bures


def wasserstein_bures_covariance(fake_phi, real_phi, epsilon=10e-14, sqrtm_func=sqrt_newton_schulz, normalize=True, dtype='float64',weight=1.,method='NewtonSchultz',adaptive_weight = 'fixed'): 
    if dtype == 'float64':
        fake_phi = tf.cast(fake_phi, tf.float64)
        real_phi = tf.cast(real_phi, tf.float64)

    batch_size = fake_phi.shape[0]
    h_dim = fake_phi.shape[1]

    # Center and normalize
    fake_phi = fake_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(fake_phi, axis=0, keepdims=True)
    real_phi = real_phi - tf.ones(shape=(batch_size, 1), dtype=dtype) @ tf.math.reduce_mean(real_phi, axis=0, keepdims=True)
    if normalize:
        fake_phi = tf.nn.l2_normalize(fake_phi, 1)
        real_phi = tf.nn.l2_normalize(real_phi,1)
    
    # bures
    C1 = tf.transpose(fake_phi) @ fake_phi
    C1 = C1 + epsilon * tf.eye(h_dim, dtype=dtype)
    C2 = tf.transpose(real_phi) @ real_phi
    C2 = C2 + epsilon * tf.eye(h_dim, dtype=dtype)
    
    if method == 'NewtonSchultz':
        bures = tf.linalg.trace(C1) + tf.linalg.trace(C2) - 2 * tf.linalg.trace(sqrtm_func(C1 @ C2))
    else:
        bures = tf.linalg.trace(C1) + tf.linalg.trace(C2) - 2 * tf.linalg.trace(matrix_sqrt_eigen(C1 @ C2))

    return weight * bures


# GDPP Loss as in https://arxiv.org/abs/1812.00068, original code adapted to Tensorflow 2
def gdpp_diversity_loss(h_fake, h_real, dtype='float64'):  # GDPP Loss
    if dtype == 'float64':
        h_fake = tf.cast(h_fake, dtypes.float64)
        h_real = tf.cast(h_real, dtypes.float64)

    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = h @ tf.transpose(h)
        eig_val, eig_vec = tf.linalg.eigh(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        # Min-max-Normalize Eig-Values
        return tf.divide(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                         tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(real_eig_val, fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss
