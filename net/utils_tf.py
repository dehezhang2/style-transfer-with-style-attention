import tensorflow as tf
import vgg
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def zca_normalization(features):
    shape = tf.shape(features)

    # reshape the features to orderless feature vectors
    mean_features = tf.reduce_mean(features, axis=[1, 2], keep_dims=True)
    unbiased_features = tf.reshape(features - mean_features, shape=(shape[0], -1, shape[3]))

    # get the covariance matrix
    gram = tf.matmul(unbiased_features, unbiased_features, transpose_a=True)
    gram /= tf.reduce_prod(tf.cast(shape[1:3], tf.float32))

    # converting the feature spaces
    s, u, v = tf.svd(gram, compute_uv=True)
    s = tf.expand_dims(s, axis=1)  # let it be active in the last dimension

    # get the effective singular values
    valid_index = tf.cast(s > 0.00001, dtype=tf.float32)
    s_effective = tf.maximum(s, 0.00001)
    sqrt_s_effective = tf.sqrt(s_effective) * valid_index
    sqrt_inv_s_effective = tf.sqrt(1.0/s_effective) * valid_index

    # colorization functions
    colorization_kernel = tf.matmul(tf.multiply(u, sqrt_s_effective), v, transpose_b=True)

    # normalized features
    normalized_features = tf.matmul(unbiased_features, u)
    normalized_features = tf.multiply(normalized_features, sqrt_inv_s_effective)
    normalized_features = tf.matmul(normalized_features, v, transpose_b=True)
    normalized_features = tf.reshape(normalized_features, shape=shape)

    return normalized_features, colorization_kernel, mean_features


def zca_colorization(normalized_features, colorization_kernel, mean_features):
    # broadcasting the tensors for matrix multiplication
    shape = tf.shape(normalized_features)
    normalized_features = tf.reshape(
        normalized_features, shape=(shape[0], -1, shape[3]))
    colorized_features = tf.matmul(normalized_features, colorization_kernel)
    colorized_features = tf.reshape(colorized_features, shape=shape) + mean_features
    return colorized_features


def adain_normalization(features):
    epsilon = 1e-7
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keep_dims=True)
    normalized_features = tf.div(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features


def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return tf.sqrt(colorization_kernels) * normalized_features + mean_features


def project_features(features, projection_module='ZCA'):
    if projection_module == 'ZCA':
        return zca_normalization(features)
    elif projection_module == 'AdaIN':
        return adain_normalization(features)
    else:
        return features, None, None


def reconstruct_features(projected_features, feature_kernels, mean_features, reconstruction_module='ZCA'):
    if reconstruction_module == 'ZCA':
        return zca_colorization(projected_features, feature_kernels, mean_features)
    elif reconstruction_module == 'AdaIN':
        return adain_colorization(projected_features, feature_kernels, mean_features)
    else:
        return projected_features


def instance_norm(inputs, epsilon=1e-10):
    inst_mean, inst_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    normalized_inputs = tf.div( tf.subtract(inputs, inst_mean), tf.sqrt(tf.add(inst_var, epsilon)))
    return normalized_inputs

def mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    num_channels = 3
    channels = tf.split(images, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=2)


def mean_image_summation(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    num_channels = 3
    channels = tf.split(image, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=2)


def batch_mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=3)


def batch_mean_image_summation(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=3)


def compute_total_variation_loss_l1(inputs, weights=1, scope=None):
    inputs_shape = tf.shape(inputs)
    height = inputs_shape[1]
    width = inputs_shape[2]

    with tf.variable_scope(scope, 'total_variation_loss', [inputs]):
        loss_y = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, height-1, -1, -1]),
            tf.slice(inputs, [0, 1, 0, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_y')
        loss_x = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, -1, width-1, -1]),
            tf.slice(inputs, [0, 0, 1, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_x')
        loss = loss_y + loss_x
        return loss
    
    
def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def k_means(image, clusters_num):
    image = tf.squeeze(image)
    print("k_means", image.shape)
    _points = tf.reshape(image, (-1, 1))
    centroids = tf.slice(tf.random_shuffle(_points), [0, 0], [clusters_num, -1])
    points_expanded = tf.expand_dims(_points, 0)

    for i in xrange(80):
        centroids_expanded = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
        assignments = tf.argmin(distances, 0)
        centroids = tf.concat(
            [tf.reduce_mean(tf.gather(_points, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c
             in
             xrange(clusters_num)], 0)

    centroids = tf.squeeze(centroids)
    centroids = -tf.nn.top_k(-centroids, clusters_num)[0]  # sort
    return centroids


if __name__ == "__main__":
    import cv2
    img = cv2.imread('lenna_cropped.jpg', cv2.IMREAD_GRAYSCALE)
    points = tf.cast(tf.convert_to_tensor(img), tf.float32)
    print(points.shape)
    centroids = k_means(points, 4)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(centroids))