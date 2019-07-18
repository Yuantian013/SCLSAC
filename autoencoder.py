

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import scipy.io as sio
import numpy as np


# Parameters
learning_rate = 0.01
training_epochs = 5

load_data = sio.loadmat('data.mat')
data=load_data['s']
load_data_all = sio.loadmat('data_all.mat')
data_all=load_data_all['s']
data=np.concatenate((data,data_all))
shape=np.shape(data)
print(shape)


class autoencoder(object):
    def __init__(self, ):
        ###############################  Model parameters  ####################################

        self.S = tf.placeholder(tf.float32, [None, 4], 's')
        self.encoder_op = self.encoder(self.S)
        self.decoder_op = self.decoder(self.encoder_op)
        self.y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        self.y_true = self.S

        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [self.optimizer]

    def learn(self, batch_xs):
        _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.S: batch_xs})
        return c

    # critic模块
    def encoder(self, s, name='Encoder', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_input = 4
            n_hidden_1 = 16
            n_hidden_2 = 8
            n_hidden_3 = 4
            n_hidden_4 = 1

            h1 = tf.get_variable('h1', [n_input, n_hidden_1], trainable=trainable)
            b1 = tf.get_variable('b1', [n_hidden_1], trainable=trainable)
            h2 = tf.get_variable('h2', [n_hidden_1, n_hidden_2], trainable=trainable)
            b2 = tf.get_variable('b2', [n_hidden_2], trainable=trainable)
            h3 = tf.get_variable('h3', [n_hidden_2, n_hidden_3], trainable=trainable)
            b3 = tf.get_variable('b3', [n_hidden_3], trainable=trainable)
            h4 = tf.get_variable('h4', [n_hidden_3, n_hidden_4], trainable=trainable)
            b4 = tf.get_variable('b4', [n_hidden_4], trainable=trainable)

            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(s,h1),b1))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, h2),b2))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, h3),b3))
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, h4),b4))
            return layer_4

    def decoder(self, encoder_op, name='Decoder', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_input = 4
            n_hidden_1 = 16
            n_hidden_2 = 8
            n_hidden_3 = 4
            n_hidden_4 = 1
            h1 = tf.get_variable('h1', [n_hidden_4, n_hidden_3], trainable=trainable)
            b1 = tf.get_variable('b1', [n_hidden_3], trainable=trainable)
            h2 = tf.get_variable('h2', [n_hidden_3, n_hidden_2], trainable=trainable)
            b2 = tf.get_variable('b2', [n_hidden_2], trainable=trainable)
            h3 = tf.get_variable('h3', [n_hidden_2, n_hidden_1], trainable=trainable)
            b3 = tf.get_variable('b3', [n_hidden_1], trainable=trainable)
            h4 = tf.get_variable('h4', [n_hidden_1, n_input], trainable=trainable)
            b4 = tf.get_variable('b4', [n_input], trainable=trainable)

            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op, h1),
                                           b1))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, h2),
                                           b2))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, h3),
                                           b3))
            layer_4 = tf.add(tf.matmul(layer_3, h4),
                             b4)
            return layer_4

    def save_result(self, path):
        save_path = self.saver.save(self.sess, path + "/model.ckpt")
        print("Save to path: ", save_path)

# Launch the graph

aec=autoencoder()
# Training cycle
for epoch in range(10):
    # Loop over all batches
    i=0
    np.random.shuffle(data)
    while i < shape[0]:
        batch_xs = data[i:i+128] # max(x) = 1, min(x) = 0
        print(batch_xs)
        # print(batch_xs)
        c=aec.learn(batch_xs)
        i=i+128
    print(c)
    aec.save_result('autoencoder')


print("Optimization Finished!")

print(aec.sess.run(aec.y_pred,feed_dict={aec.S:data[0:5]}),data[0:5])



