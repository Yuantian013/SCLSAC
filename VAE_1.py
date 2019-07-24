

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import scipy.io as sio
import numpy as np
import tensorflow_probability as tfp

# Parameters
learning_rate = 0.0001
training_epochs = 5

load_data = sio.loadmat('data.mat')
data=load_data['s']
load_data_all = sio.loadmat('data_all.mat')
data_all=load_data_all['s']
data=np.concatenate((data,data_all))
shape=np.shape(data)
print(shape)


class VAE(object):
    def __init__(self, ):
        tf.reset_default_graph()
        ###############################  Model parameters  ####################################

        self.X_dim = 4
        self.z_dim = 32
        self.h_dim = 16

        self.S = tf.placeholder(tf.float32, [None, self.X_dim], 's')
        self.Z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        self.z_mu, self.z_logvar, distribution= self.encoder(self.S)



        # Sampling latent after encoding
        # z_sample = distribution.sample(self.num_of_z_samples)
        z_sample = self.sample_z(self.z_mu, self.z_logvar)

        x_mu, self.output_distribution = self.decoder(z_sample)

        self.reconstruct_x, _ = self.decoder(self.z_mu, reuse=True)

        self.recon_loss = tf.reduce_mean(tf.pow(x_mu - self.S , 2))

        # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
        self.kl_loss = 0.5 * tf.reduce_mean(tf.exp(self.z_logvar) + self.z_mu ** 2 - 1. - self.z_logvar,1)

        # self.kl_loss = self.wasserstein_distance(z_mu,z_logvar,mu,logvar)
        # VAE loss
        self.vae_loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
        self.optimizer = tf.train.AdamOptimizer(0.00001).minimize(self.vae_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [self.optimizer]

    def learn(self, batch_xs):
        _, vae_loss,kl_loss,recon_loss = self.sess.run([self.optimizer, self.vae_loss,self.kl_loss,self.recon_loss], feed_dict={self.S: batch_xs})
        return vae_loss,kl_loss,recon_loss

    # =============================== Q(z|X) ======================================
    # critic模块
    def encoder(self, s, name='Encoder', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            Q_W1 = tf.Variable(self.xavier_init([self.X_dim, self.h_dim]),trainable=trainable)
            Q_b1 = tf.Variable(tf.zeros(shape=[self.h_dim]),trainable=trainable)
            Q_W2_mu = tf.Variable(self.xavier_init([self.h_dim, self.z_dim]),trainable=trainable)
            Q_b2_mu = tf.Variable(tf.zeros(shape=[self.z_dim]),trainable=trainable)

            Q_W2_sigma = tf.Variable(self.xavier_init([self.h_dim, self.z_dim]),trainable=trainable)
            Q_b2_sigma = tf.Variable(tf.zeros(shape=[self.z_dim]),trainable=trainable)
            h = tf.nn.sigmoid(tf.matmul(s, Q_W1) + Q_b1)
            z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
            z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
            distribution = tfp.distributions.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.exp(z_logvar))
            return z_mu, z_logvar,distribution

    # =============================== P(X|z) ======================================
    def decoder(self, z, name='Decoder', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(z, self.h_dim, activation=tf.nn.sigmoid, name='l1', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_0, self.X_dim, activation=None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_0, self.X_dim, None, trainable=trainable)
            distribution = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))
            return mu, distribution


    def save_result(self, path):
        save_path = self.saver.save(self.sess, path + "/model.ckpt")
        print("Save to path: ", save_path)

    def xavier_init(self,size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def sample_z(self, mu, log_var):
        base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.z_dim),
                                                                     scale_diag=tf.ones(self.z_dim))
        eps = base_distribution.sample(tf.shape(mu)[0])

        return mu + tf.exp(log_var) * eps

    def load_result(self):
        self.saver.restore(self.sess, "VAE/model.ckpt")  # 1 0.1 0.5 0.001

    def encode(self,s):
        return self.sess.run([self.z_mu, self.z_logvar], {self.S: s})

# Launch the graph

vae=VAE()
vae.load_result()
np.random.shuffle(data)
# # Training cycle
# for epoch in range(100):
#     # Loop over all batches
#     i=0
#     np.random.shuffle(data)
#     while i < shape[0]-200:
#         batch_xs = data[i:i+128] # max(x) = 1, min(x) = 0
#         vae_loss,kl_loss,recon_loss=vae.learn(batch_xs)
#         i=i+128
#     print(vae_loss,np.mean(kl_loss),np.mean(recon_loss))
#     # print(vae.sess.run(vae.logits, feed_dict={vae.S: data[0:1]}), data[0:1])
# vae.save_result('VAE')
#
#
# print("Optimization Finished!")
print(data[0:5])
print(vae.sess.run(vae.z_mu,feed_dict={vae.S:data[0:5]}))



