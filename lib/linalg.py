import tensorflow as tf


# Matrix square root using the Newton-Schulz method
def sqrt_newton_schulz(A, iterations=15, dtype='float64'):
    dim = A.shape[0]
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


# Matrix square root using the eigenvalue decompostion
def matrix_sqrt_eigen(mat):
    eig_val, eig_vec = tf.linalg.eigh(mat)
    diagonal = tf.linalg.diag(tf.pow(eig_val, 0.5))
    mat_sqrt = tf.matmul(diagonal, tf.transpose(eig_vec))
    mat_sqrt = tf.matmul(eig_vec, mat_sqrt)
    return mat_sqrt


# Calculates the error, as the absolute difference in norms of a matrix A and the reconstructed A = Asqrt @ Asqrt
def error(Asqrt, A):
    Ar = Asqrt @ Asqrt
    return tf.abs(tf.linalg.norm(Ar) - tf.linalg.norm(A))


# mean-square-and-cross-product) matrix
def MSCP_matrix(A):
    n = A.shape[0]
    if A.dtype == 'float64':
       n = tf.cast(n, tf.float64)
    C = (tf.transpose(A) @ A) / n
    return C

def scatter_matrix(A):
    A = A - tf.ones(shape=(A.shape[0], 1), dtype=A.dtype) @ tf.math.reduce_mean(A, axis=0, keepdims=True)
    C = tf.transpose(A) @ A
    return C

# covariance matrix = scatter matrix / (n-1)
def covariance_matrix(A):
    n = A.shape[0]
    if A.dtype == 'float64':
        n = tf.cast(n, tf.float64)
    C = scatter_matrix(A) / (n-1)
    return C

def correlation_matrix(A):
    n = A.shape[0]
    if A.dtype == 'float64':
        n = tf.cast(n, tf.float64)
    A = A - tf.ones(shape=(n, 1), dtype=A.dtype) @ tf.math.reduce_mean(A, axis=0, keepdims=True)
    A = A / (tf.ones(shape=(n, 1), dtype=A.dtype) @ tf.math.reduce_std(A, axis=0, keepdims=True))
    C = (tf.transpose(A) @ A) / n
    return C

def cosine_similarity_matrix(A):
    A = tf.nn.l2_normalize(A, 0)
    C = tf.transpose(A) @ A
    return C

def centered_unit_circle_projected_matrix(A):
    n = A.shape[0]
    if A.dtype == 'float64':
        n = tf.cast(n, tf.float64)
    A = A - tf.ones(shape=(n, 1), dtype=A.dtype) @ tf.math.reduce_mean(A, axis=0, keepdims=True)
    A = tf.nn.l2_normalize(A, 1)
    C = tf.transpose(A) @ A
    return C