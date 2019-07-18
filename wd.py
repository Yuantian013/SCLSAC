
import tensorflow as tf
import numpy as np
import time
from SAC.squash_bijector import SquashBijector
import tensorflow_probability as tfp

# distribution=tfp.distributions.MultivariateNormalDiag(loc=tf.constant([4.5,1.,1.,1.,1.]), scale_diag=tf.constant([0.1,0.1,0.1,0.1,0.1]))
# sess=tf.Session()
# print(sess.run(distribution.covariance()))
#
# #取tf mu和sigma
# u1=sess.run(distribution.loc)
#
# o1=[0.1,0.1,0.1,0.1,0.1]
# u2=np.array([5,1.,1.,1.,1.])
# o2=[0.1,0.1,0.1,0.1,0.1]
#
# #得到对角阵sigma
# o_1=np.eye(np.size(o1))*o1*o1
# o_2=np.eye(np.size(o1))*o2*o2
# print(o_1)
# #W 2(P, Q) = ||μ1 − μ2||2 + B2(Σ1, Σ2)
# # B2(Σ , Σ2) = tr(Σ ) + tr(Σ ) − 2tr
#


u1=np.array([0.45])
u2=np.array([0.5])
o_1=np.array([[0.01]])
o_2=np.array([[0.01]])
dis_part1=np.linalg.norm(u1-u2) **2
dis_part2=np.trace(o_1+o_2-2*np.sqrt(np.dot(np.dot(np.sqrt(o_2),o_1),np.sqrt(o_2))))
print(dis_part1+dis_part2)

