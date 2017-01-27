# imports
import tensorflow as tf
import numpy as np

# functions
def max_pool(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def deconv2d(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

###############
# batch norm layers
# batch normalization : deals with poor initialization helps gradient flow
d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')

g_bn_e1a = batch_norm(name='g_bn_e1a')
g_bn_e2a = batch_norm(name='g_bn_e2a')
g_bn_e3a = batch_norm(name='g_bn_e3a')
g_bn_e4a = batch_norm(name='g_bn_e4a')
g_bn_e5a = batch_norm(name='g_bn_e5a')

g_bn_e1b = batch_norm(name='g_bn_e1b')
g_bn_e2b = batch_norm(name='g_bn_e2b')
g_bn_e3b = batch_norm(name='g_bn_e3b')
g_bn_e4b = batch_norm(name='g_bn_e4b')
g_bn_e5b = batch_norm(name='g_bn_e5b')

g_bn_d1a = batch_norm(name='g_bn_d1a')
g_bn_d2a = batch_norm(name='g_bn_d2a')
g_bn_d3a = batch_norm(name='g_bn_d3a')
g_bn_d4a = batch_norm(name='g_bn_d4a')

g_bn_d1b = batch_norm(name='g_bn_d1b')
g_bn_d2b = batch_norm(name='g_bn_d2b')
g_bn_d3b = batch_norm(name='g_bn_d3b')
g_bn_d4b = batch_norm(name='g_bn_d4b')

g_bn_d1c = batch_norm(name='g_bn_d1c')
g_bn_d2c = batch_norm(name='g_bn_d2c')
g_bn_d3c = batch_norm(name='g_bn_d3c')
g_bn_d4c = batch_norm(name='g_bn_d4c')


batch_size = 1
x = tf.placeholder(tf.float32,[batch_size,256,256,1])
y = tf.placeholder(tf.float32,[batch_size,256,256,1])

e1_a = conv2d(x, 64, name='g_e1_conv_a')
e1_b = g_bn_e1a(conv2d(lrelu(e1_a), 64, name='g_e1_conv_b'))
e1_c = g_bn_e1b(conv2d(lrelu(e1_b), 64, name='g_e1_conv_c'))

m1 = max_pool(e1_c)

e2_a = g_bn_e2a(conv2d(lrelu(m1), 128, name='g_e2_conv_a'))
e2_b = g_bn_e2b(conv2d(lrelu(e2_a), 128, name='g_e2_conv_b'))

m2 = max_pool(e2_b)

e3_a = g_bn_e3a(conv2d(lrelu(m2), 256, name='g_e3_conv_a'))
e3_b = g_bn_e3b(conv2d(lrelu(e3_a), 256, name='g_e3_conv_b'))

m3 = max_pool(e3_b)

e4_a = g_bn_e4a(conv2d(lrelu(m3), 512, name='g_e4_conv_a'))
e4_b = g_bn_e4b(conv2d(lrelu(e4_a), 512, name='g_e4_conv_b'))

m4 = max_pool(e4_b)

e5_a = g_bn_e5a(conv2d(lrelu(m4), 1024, name='g_e5_conv_a'))
e5_b = g_bn_e5b(conv2d(lrelu(e5_a), 1024, name='g_e5_conv_b'))

d1, d1_w, d1_b = deconv2d(lrelu(e5_b),[batch_size, 32, 32, 512], name='g_d1', with_w=True)
d1 = g_bn_d1a(d1)
d1 = tf.concat([d1, e4_b],3)

d1_a = g_bn_d1b(conv2d(lrelu(d1), 512, name='g_d1_conv_a'))
d1_b = g_bn_d1c(conv2d(lrelu(d1_a), 512, name='g_d1_conv_b'))

d2, d2_w, d2_b = deconv2d(lrelu(d1_b),[batch_size, 64, 64, 256], name='g_d2', with_w=True)
d2 = g_bn_d2a(d2)
d2 = tf.concat([d2, e3_b],3)

d2_a = g_bn_d2b(conv2d(lrelu(d2), 256, name='g_d2_conv_a'))
d2_b = g_bn_d2c(conv2d(lrelu(d2_a), 256, name='g_d2_conv_b'))

d3, d3_w, d3_b = deconv2d(lrelu(d2_b),[batch_size, 128, 128, 128], name='g_d3', with_w=True)
d3 = g_bn_d3a(d3)
d3 = tf.concat([d3, e2_b],3)

d3_a = g_bn_d3b(conv2d(lrelu(d3), 128, name='g_d3_conv_a'))
d3_b = g_bn_d3c(conv2d(lrelu(d3_a), 128, name='g_d3_conv_b'))

d4, d4_w, d4_b = deconv2d(lrelu(d3_b),[batch_size, 256, 256, 64], name='g_d4', with_w=True)
d4 = g_bn_d4a(d4)
d4 = tf.concat([d4, e1_b],3)

d4_a = g_bn_d4b(conv2d(lrelu(d4), 64, name='g_d4_conv_a'))
d4_b = g_bn_d4c(conv2d(lrelu(d4_a), 64, name='g_d4_conv_b'))

resid = conv2d(d4_b, 1, k_h=1, k_w=1, name='residual')

out = tf.add(resid,x,name='out')

loss = tf.nn.l2_loss(out-y)
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

import IPython; IPython.embed()
