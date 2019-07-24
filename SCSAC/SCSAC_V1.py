
# class autoencoder(object):
#     def __init__(self, ):
#         ###############################  Model parameters  ####################################
#
#         self.S = tf.placeholder(tf.float32, [None, 4], 's')
#         self.encoder_op = self.encoder(self.S)
#         self.decoder_op = self.decoder(self.encoder_op)
#         self.y_pred = self.decoder_op
#         # Targets (Labels) are the input data.
#         self.y_true = self.S
#
#         self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
#         self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
#
#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())
#         self.saver = tf.train.Saver()
#         self.opt = [self.optimizer]
#
#     def learn(self, batch_xs):
#         _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.S: batch_xs})
#         return c
#
#     # critic模块
#     def encoder(self, s, name='Encoder', reuse=None, custom_getter=None):
#         trainable = True if reuse is None else False
#         with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
#             n_input = 4
#             n_hidden_1 = 16
#             n_hidden_2 = 8
#             n_hidden_3 = 4
#             n_hidden_4 = 1
#
#             h1 = tf.get_variable('h1', [n_input, n_hidden_1], trainable=trainable)
#             b1 = tf.get_variable('b1', [n_hidden_1], trainable=trainable)
#             h2 = tf.get_variable('h2', [n_hidden_1, n_hidden_2], trainable=trainable)
#             b2 = tf.get_variable('b2', [n_hidden_2], trainable=trainable)
#             h3 = tf.get_variable('h3', [n_hidden_2, n_hidden_3], trainable=trainable)
#             b3 = tf.get_variable('b3', [n_hidden_3], trainable=trainable)
#             h4 = tf.get_variable('h4', [n_hidden_3, n_hidden_4], trainable=trainable)
#             b4 = tf.get_variable('b4', [n_hidden_4], trainable=trainable)
#
#             layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(s,h1),b1))
#             layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, h2),b2))
#             layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, h3),b3))
#             layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, h4),b4))
#             return layer_4
#
#     def decoder(self, encoder_op, name='Decoder', reuse=None, custom_getter=None):
#         trainable = True if reuse is None else False
#         with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
#             n_input = 4
#             n_hidden_1 = 16
#             n_hidden_2 = 8
#             n_hidden_3 = 4
#             n_hidden_4 = 1
#             h1 = tf.get_variable('h1', [n_hidden_4, n_hidden_3], trainable=trainable)
#             b1 = tf.get_variable('b1', [n_hidden_3], trainable=trainable)
#             h2 = tf.get_variable('h2', [n_hidden_3, n_hidden_2], trainable=trainable)
#             b2 = tf.get_variable('b2', [n_hidden_2], trainable=trainable)
#             h3 = tf.get_variable('h3', [n_hidden_2, n_hidden_1], trainable=trainable)
#             b3 = tf.get_variable('b3', [n_hidden_1], trainable=trainable)
#             h4 = tf.get_variable('h4', [n_hidden_1, n_input], trainable=trainable)
#             b4 = tf.get_variable('b4', [n_input], trainable=trainable)
#
#             layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op, h1),
#                                            b1))
#             layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, h2),
#                                            b2))
#             layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, h3),
#                                            b3))
#             layer_4 = tf.add(tf.matmul(layer_3, h4),
#                              b4)
#             return layer_4
#
#     def encode(self,s):
#
#         return self.sess.run(self.encoder_op, {self.S: s})[0]
#
#
#     def save_result(self, path):
#         save_path = self.saver.save(self.sess, path + "/model.ckpt")
#         print("Save to path: ", save_path)
#     def load_result(self):
#         self.saver.restore(self.sess, "autoencoder/model.ckpt")  # 1 0.1 0.5 0.001

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


import tensorflow as tf
import numpy as np
import time
from SAC.squash_bijector import SquashBijector
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
from variant import VARIANT, get_env_from_name, get_policy, get_train
from .utils import get_evaluation_rollouts, evaluate_rollouts, evaluate_training_rollouts
import logger
from safety_constraints import get_safety_constraint_func

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 50)


SELF_LEARN=True
class SCSAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 vae_dim,
                 variant,

                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        tf.reset_default_graph()
        self.memory_capacity = variant['memory_capacity']
        self.cons_memory_capacity = variant['cons_memory_capacity']
        self.batch_size = variant['batch_size']
        gamma = variant['gamma']
        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        self.max_grad_norm = variant['max_grad_norm']
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 3), dtype=np.float32)
        self.cons_memory = np.zeros((self.cons_memory_capacity, s_dim * 2 + a_dim + 3+vae_dim), dtype=np.float32)
        self.pointer = 0
        self.cons_pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        self.a_dim, self.s_dim,self.vae_dim = a_dim, s_dim,vae_dim
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim  #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.s_aec = tf.placeholder(tf.float32, [None, self.vae_dim ], 's_aec')
        self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.cons_S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
        self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.l_R = tf.placeholder(tf.float32, [None, 1], 'l_r')  # 给lyapunov设计的reward
        self.cons_l_R = tf.placeholder(tf.float32, [None, 1], 'cons_l_r')
        self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
        # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
        labda = variant['labda']
        alpha = variant['alpha']
        alpha3 = variant['alpha3']
        log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
        log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature
        self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
        self.alpha = tf.exp(log_alpha)

        self.a, self.deterministic_a, self.a_dist = self._build_a(self.S, )  # 这个网络用于及时更新参数
        self.distribution = self._build_d()
        self.mu = self.distribution.loc
        self.sigma = self.distribution.covariance()
        self.q1 = self._build_c(self.S, self.a_input, 'critic1')  # 这个网络是用于及时更新参数
        self.q2 = self._build_c(self.S, self.a_input, 'critic2')  # 这个网络是用于及时更新参数
        self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络

        self.q1_a = self._build_c(self.S, self.a, 'critic1', reuse=True)
        self.q2_a = self._build_c(self.S, self.a, 'critic2', reuse=True)

        self.use_lyapunov = variant['use_lyapunov']
        self.adaptive_alpha = variant['adaptive_alpha']

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic1')
        c2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic2')
        l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Distribution')

        ###############################  Model Learning Setting  ####################################
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        target_update = [ema.apply(a_params), ema.apply(c1_params),ema.apply(c2_params), ema.apply(l_params),ema.apply(d_params)]  # soft update operation

        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters

        cons_a, _, cons_a_dist = self._build_a(self.cons_S, reuse=True)
        cons_a_, _, cons_a_dist_ = self._build_a(self.cons_S_, reuse=True)
        self.cons_a_input = tf.placeholder(tf.float32, [None, a_dim], 'cons_a_input')
        # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
        # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
        self.log_pis = log_pis = self.a_dist.log_prob(self.a)
        self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q1_ = self._build_c(self.S_, a_,'critic1', reuse=True, custom_getter=ema_getter)
        q2_ = self._build_c(self.S_, a_, 'critic2', reuse=True, custom_getter=ema_getter)
        l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
        self.cons_l = self._build_l(self.cons_S, self.cons_a_input, reuse=True)
        self.cons_l_ = self._build_l(self.cons_S_, cons_a_, reuse=True)

        # lyapunov constraint

        self.l_derta = tf.reduce_mean(self.cons_l_ - self.cons_l + alpha3 * self.cons_l_R)

        labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
        self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
        self.lambda_train = tf.train.AdamOptimizer(self.LR_A).minimize(labda_loss, var_list=log_labda)
        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim),
                scale_diag=tf.ones(self.a_dim))
            policy_prior_log_probs = policy_prior.log_prob(self.a)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        min_Q_target = tf.reduce_min((self.q1_a, self.q2_a), axis=0)
        self.a_preloss = a_preloss = tf.reduce_mean(self.alpha * log_pis - min_Q_target - policy_prior_log_probs)


        if self.use_lyapunov is True:
            a_loss = self.labda * self.l_derta + a_preloss
        else:
            a_loss = a_preloss
        self.a_loss = a_loss
        self.trainer = tf.train.AdamOptimizer(self.LR_A)
        pre_grads_and_var = self.trainer.compute_gradients(a_preloss, a_params)

        pre_grads, pre_var = zip(*pre_grads_and_var)
        # self.pre_max_grad = tf.reduce_max(tf.concat([tf.reshape(grad, [-1]) for grad in pre_grads], axis=0))

        grads_and_var = self.trainer.compute_gradients(a_loss, a_params)
        grads, var = zip(*grads_and_var)
        # self.max_grad = tf.reduce_max(tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0))

        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            pre_grads, _grad_norm = tf.clip_by_global_norm(pre_grads, self.max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        grads_and_var = list(zip(grads, var))
        pre_grads_and_var = list(zip(pre_grads, pre_var))

        self.atrain = self.trainer.apply_gradients(grads_and_var) #以learning_rate去训练，方向是minimize loss，调整列表参数，用adam
        self.a_pretrain = self.trainer.apply_gradients(pre_grads_and_var)
        next_log_pis = a_dist_.log_prob(a_)
        with tf.control_dependencies(target_update):  # soft replacement happened at here
            min_next_q = tf.reduce_min([q1_,q2_],axis=0)
            q1_target = self.R + gamma * (1-self.terminal) * tf.stop_gradient(min_next_q - self.alpha * next_log_pis)    #ddpg
            q2_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(min_next_q - self.alpha * next_log_pis)  # ddpg
            if self.approx_value:
                l_target = self.l_R + gamma * (1-self.terminal)*l_   # Lyapunov critic - self.alpha * next_log_pis
            else:
                l_target = self.l_R

            # self.distribution_loss = -tf.reduce_mean(self.distribution.prob(self.s_aec))
            self.distribution_loss = -tf.reduce_mean(self.distribution.log_prob(self.s_aec))


            self.d_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.c1_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.c2_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.l_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.td_error1 = tf.losses.mean_squared_error(labels=q1_target, predictions=self.q1)
            self.td_error2 = tf.losses.mean_squared_error(labels=q2_target, predictions=self.q2)
            self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
            c1_grads_and_var = self.c1_trainer.compute_gradients(self.td_error1, c1_params)
            c2_grads_and_var = self.c2_trainer.compute_gradients(self.td_error2, c2_params)
            l_grads_and_var = self.l_trainer.compute_gradients(self.l_error, l_params)
            d_grads_and_var = self.d_trainer.compute_gradients(self.distribution_loss, d_params)

            c1_grads, c1_var = zip(*c1_grads_and_var)
            c2_grads, c2_var = zip(*c2_grads_and_var)
            l_grads, l_var = zip(*l_grads_and_var)
            d_grads, d_var = zip(*d_grads_and_var)

            c1_grads_and_var = list(zip(c1_grads, c1_var))
            c2_grads_and_var = list(zip(c2_grads, c2_var))
            l_grads_and_var = list(zip(l_grads, l_var))
            d_grads_and_var = list(zip(d_grads, d_var))

            self.ctrain1 = self.c1_trainer.apply_gradients(c1_grads_and_var)
            self.ctrain2 = self.c2_trainer.apply_gradients(c2_grads_and_var)
            self.ltrain = self.l_trainer.apply_gradients(l_grads_and_var)
            self.dtrain = self.d_trainer.apply_gradients(d_grads_and_var)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [ self.ctrain1, self.ctrain2, ]
        self.diag_names = ['labda', 'alpha', 'critic1_error','critic2_error','lyapunov_error', 'entropy', 'policy_loss','distribution_loss']
        self.diagnotics = [self.labda, self.alpha, self.td_error1, self.td_error2, self.l_error, tf.reduce_mean(-self.log_pis)]

        if self.adaptive_alpha is True:
            self.opt.append(self.alpha_train)
        if self.use_lyapunov is True:
            self.opt.extend([self.ltrain, self.lambda_train])


    def choose_action(self, s, evaluation = False):
        if evaluation is True:
            return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def get_distribution(self):
        return self.sess.run(self.mu)[0], self.sess.run(self.sigma)[0]


    def learn(self, LR_A, LR_C, LR_L):
        if self.pointer>=self.memory_capacity:
            indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]  # state
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  # action
        br = bt[:, -self.s_dim - 3: -self.s_dim - 2]  # reward
        blr = bt[:, -self.s_dim - 2: -self.s_dim -1]  # l_reward
        bterminal = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]  # next state

        feed_dict = {self.a_input: ba, self.S: bs, self.S_: bs_, self.R: br, self.l_R: blr, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L}

        if self.use_lyapunov is True:
            # 边缘的 s a s_ l_r
            if self.cons_pointer <= self.batch_size:
                opt_list = self.opt[:-1] + [self.a_pretrain]
                diagnotics = self.diagnotics + [self.a_preloss]
            else:
                if self.cons_pointer >= self.cons_memory_capacity:
                    indices = np.random.choice(self.cons_memory_capacity, size=self.batch_size)
                else:
                    indices = np.random.choice(self.cons_pointer, size=self.batch_size)

                bt = self.cons_memory[indices, :]
                cons_bs = bt[:, :self.s_dim]
                cons_ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
                cons_bs_ = bt[:, self.s_dim + self.a_dim+3:2*self.s_dim + self.a_dim+3]
                cons_blr = bt[:, self.s_dim + self.a_dim+1:  self.s_dim + self.a_dim+2]
                cons_bsaec= bt[:, 2*self.s_dim + self.a_dim+3:  2*self.s_dim + self.a_dim+3+self.vae_dim]

                feed_dict.update({self.cons_a_input: cons_ba, self.cons_S: cons_bs, self.cons_S_: cons_bs_,
                                  self.cons_l_R: cons_blr,self.s_aec:cons_bsaec})

                opt_list = self.opt + [self.atrain]   + [self.dtrain]
                diagnotics = self.diagnotics + [self.a_loss] + [self.distribution_loss]
        else:
            opt_list = self.opt + [self.atrain]
            diagnotics = self.diagnotics + [ self.a_loss]


        self.sess.run(opt_list, feed_dict)
        diagnotics = self.sess.run(diagnotics, feed_dict)

        return diagnotics

    def store_transition(self, s, a, r, l_r, terminal, s_):
        transition = np.hstack((s, a, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def store_edge_transition(self, s, a,r, l_r, terminal, s_,s_aec):
        """把数据存入constraint buffer"""
        transition = np.hstack((s, a,[r], [l_r], [terminal], s_,s_aec))
        index = self.cons_pointer % self.cons_memory_capacity  # replace the old memory with new memory
        self.cons_memory[index, :] = transition
        self.cons_pointer += 1

    #action 选择模块也是actor模块


    def _build_a(self, s, name='Actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 64, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 64, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_1, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)


            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=mu,
                    scale_diag=sigma),
            ))
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector)

            clipped_mu = squash_bijector.forward(mu)


        return clipped_a, clipped_mu, distribution



    #critic模块
    def _build_c(self, s, a, name ='Critic', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_d(self, name='Distribution', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            # x_a = tf.get_variable('x_a',[1, self.s_dim+self.a_dim] , trainable=trainable)
            # x_ade = tf.get_variable('x_a', initializer=tf.constant([[0.5]], dtype=tf.float32), trainable=trainable)
            x_ade = tf.get_variable('x_a',[1,32] , trainable=trainable)
            sig = tf.get_variable('sig', [1,32], trainable=trainable)
            distribution = tfp.distributions.MultivariateNormalDiag(loc=x_ade, scale_diag=sig)
            return distribution
    # def _build_sample(self, name='sample', reuse=None, custom_getter=None):
    #     trainable = True if reuse is None else False
    #     with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
    #         # x_a = tf.get_variable('x_a',[1, self.s_dim+self.a_dim] , trainable=trainable)
    #         x_ade = tf.get_variable('x_a', initializer=tf.constant([[0.5]], dtype=tf.float32), trainable=trainable)
    #         # x_ade = tf.get_variable('x_a',[1, 1] , trainable=trainable)
    #         sig = tf.get_variable('sig', [1, 1], trainable=trainable)
    #         distribution = tfp.distributions.MultivariateNormalDiag(loc=x_ade, scale_diag=sig)
    #         return distribution


    def save_result(self, path):

        save_path = self.saver.save(self.sess, path+"/model.ckpt")
        print("Save to path: ", save_path)
def wasserstein_distance(m1,o1,m2,o2):
    o1=np.array(o1)
    o2=np.array(o2)
    dis_part1 = np.linalg.norm(m1 - m2) ** 2
    dis_part2 = np.trace(o1 + o2 - 2 * np.sqrt(np.dot(np.dot(np.sqrt(o2), o1), np.sqrt(o2))))
    return dis_part1+dis_part2
def train(variant):
    auto_low = 0.01
    # aec = autoencoder()
    # aec.load_result()
    aec = VAE()
    aec.load_result()
    env_name = variant['env_name']
    env = get_env_from_name(env_name)
    if variant['evaluate'] is True:
        evaluation_env = get_env_from_name(env_name)
    else:
        evaluation_env = None
    env_params = variant['env_params']
    judge_safety_func = get_safety_constraint_func(variant)

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']
    num_of_paths = variant['num_of_paths']

    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])

    logger.logkv('tau', policy_params['tau'])
    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = policy_build_fn(a_dim, s_dim,32, policy_params)
    logger.logkv('target_entropy', policy.target_entropy)
    # For analyse

    Render = env_params['eval_render']
    ewma_p = 0.95
    ewma_step = np.zeros((1, max_episodes + 1))
    ewma_reward = np.zeros((1, max_episodes + 1))

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False
    for i in range(max_episodes):
        awd=0
        ep_reward = 0
        l_r = 0

        current_path = {'rewards': [],
                        'l_rewards': [],
                        'violation':[],
                        }
        [current_path.update({key:[]}) for key in policy.diag_names]
        if global_step > max_global_steps:
            break

        s = env.reset()
        s_bad_for_autoencoder=[s.tolist()]
        s_normal_for_autoencoder=[s.tolist()]
        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            # Run in simulator
            s_, r, done, info = env.step(action)
            if training_started:
                global_step += 1

            mu_bad, sigma_bad = policy.get_distribution()
            # mu_now=np.concatenate((s,a))
            mu_now,sigma_now= aec.encode([s_])

            mu_now=mu_now[0]
            sigma_now=np.eye(32)* ( np.exp(sigma_now[0]) **2)
            # print(np.shape(mu_now),np.shape(mu_bad),np.shape(sigma_now),np.shape(sigma_bad))
            # print(mu_now,[[sigma_now]],mu_bad,sigma_bad)
            wd = wasserstein_distance(mu_now, sigma_now, mu_bad, sigma_bad)  #
            awd=awd+wd
            # if i>100:
            #     print(wd)
            l_r = 100*max(30/ wd - 1, 0)

            if not SELF_LEARN:
                l_r = info['l_rewards']
                # l_r = 0

            if j == max_ep_steps - 1:
                done = True
            terminal = 1. if done else 0.

            violation_of_constraint = info['violation_of_constraint']
            # 储存s,a和s_next,reward用于DDPG的学习
            policy.store_transition(s, a, r, l_r, terminal, s_)
            s_normal_for_autoencoder.append(s_.tolist())
            # 如果状态接近边缘 就存储到边缘memory里
            # if policy.use_lyapunov is True and np.abs(s[0]) > env.cons_pos:  # or np.abs(s[2]) > env.theta_threshold_radians*0.8
            if policy.use_lyapunov is True and judge_safety_func(s_, r, done, info):  # or np.abs(s[2]) > env.theta_threshold_radians*0.8
                s_aec,_ = aec.encode([s_])
                s_bad_for_autoencoder.append(s_.tolist())
                policy.store_edge_transition(s, a, r, l_r, terminal, s_,s_aec[0])

            # Learn
            if policy.use_lyapunov is True:
                if policy.pointer > min_memory_size and global_step % steps_per_cycle == 0:
                    # Decay the action randomness
                    training_started = True
                    for _ in range(train_per_cycle):
                        train_diagnotic = policy.learn(lr_a_now, lr_c_now, lr_l_now)

            else:
                if policy.pointer > min_memory_size and global_step % steps_per_cycle == 0:
                    # Decay the action randomness
                    training_started = True
                    for _ in range(train_per_cycle):
                        train_diagnotic = policy.learn(lr_a_now, lr_c_now, lr_l_now)

            if training_started:
                current_path['rewards'].append(r)
                current_path['l_rewards'].append(l_r)
                current_path['violation'].append(violation_of_constraint)
                [current_path[key].append(value) for key,value in zip(policy.diag_names, train_diagnotic)]

            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:
                if evaluation_env is not None:
                    rollouts = get_evaluation_rollouts(policy, evaluation_env, num_of_paths, max_ep_steps,render=Render)
                    diagnotic = evaluate_rollouts(rollouts)
                    # [diagnotics[key].append(diagnotic[key]) for key in diagnotic.keys()]
                    print('training_step:', global_step, 'average eval reward:', diagnotic['return-average'],
                          'average eval lreward:', diagnotic['lreturn-average'],
                          'average eval violations:', diagnotic['violation-avg'],
                          'average length:', diagnotic['episode-length-avg'], )
                    logger.logkv('eval_eprewmean', diagnotic['return-average'])
                    logger.logkv('eval_eplrewmean', diagnotic['lreturn-average'])
                    logger.logkv('eval_eplenmean', diagnotic['episode-length-avg'])
                    logger.logkv('eval_violation_times', diagnotic['violation-avg'])
                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    # [training_diagnotics[key].append(training_diagnotic[key]) for key in training_diagnotic.keys()]\
                    logger.logkv('eprewmean', training_diagnotic['rewards'])
                    logger.logkv('eplrewmean', training_diagnotic['l_rewards'])
                    logger.logkv('eplenmean', training_diagnotic['len'])
                    logger.logkv('end_cost', training_diagnotic['end_cost'])
                    [logger.logkv(key, training_diagnotic[key]) for key in policy.diag_names]

                    logger.logkv('violation_times', training_diagnotic['violation'])
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    print('training_step:', global_step,
                          'average reward:', round(training_diagnotic['rewards'], 2),
                          'average lreward:', round(training_diagnotic['l_rewards'], 2),
                          'average violations:', training_diagnotic['violation'],
                          'end cost:', round(training_diagnotic['end_cost'],2),
                          'average length:', round(training_diagnotic['len'], 1),
                          'lyapunov error:', round(training_diagnotic['lyapunov_error'], 6),
                          'critic1 error:', round(training_diagnotic['critic1_error'], 6),
                          'critic2 error:', round(training_diagnotic['critic2_error'], 6),
                          'policy_loss:', round(training_diagnotic['policy_loss'], 6),
                          'alpha:', round(training_diagnotic['alpha'], 6),
                          'lambda:', round(training_diagnotic['labda'], 6),
                          'entropy:', round(training_diagnotic['entropy'], 6),
                          'distribution_loss', training_diagnotic['distribution_loss'],
                          'Average WD',awd/(j+1)
                          )
                          # 'max_grad:', round(training_diagnotic['max_grad'], 6)
                logger.dumpkvs()
            # 状态更新
            s = s_
            ep_reward += r

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                ewma_step[0, i + 1] = ewma_p * ewma_step[0, i] + (1 - ewma_p) * j
                ewma_reward[0, i + 1] = ewma_p * ewma_reward[0, i] + (1 - ewma_p) * ep_reward
                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic

                s_normal_for_autoencoder=s_normal_for_autoencoder[0:np.shape(s_bad_for_autoencoder)[0]]

                s_for_autoencoder=np.concatenate((s_bad_for_autoencoder,s_normal_for_autoencoder))

                c,_,_ = aec.learn(s_for_autoencoder)
                if c < auto_low:
                    auto_low = c
                    print(c)
                    aec.save_result('autoencoder')
                break
    policy.save_result(log_path)
    print('Running time: ', time.time() - t1)
    return

