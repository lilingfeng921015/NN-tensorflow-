import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
# 从X这个32行2列的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0
# 作为输入数据集的标签（正确答案）
X = rdm.rand(32, 2)
Y_ = [[int((x0 + x1) < 1)] for (x0, x1) in X]
print("X:\n", X)
print("Y_:\n", Y_)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))

    # 训练模型
    step = 3000
    for i in range(step):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y_})
            print("After %d training steps, loss_mse on all data is %g" % (i, total_loss))
    print('w1:\n', sess.run(w1))
    print('w2:\n', sess.run(w2))

'''
X:
 [[ 0.83494319  0.11482951]
 [ 0.66899751  0.46594987]
 [ 0.60181666  0.58838408]
 [ 0.31836656  0.20502072]
 [ 0.87043944  0.02679395]
 [ 0.41539811  0.43938369]
 [ 0.68635684  0.24833404]
 [ 0.97315228  0.68541849]
 [ 0.03081617  0.89479913]
 [ 0.24665715  0.28584862]
 [ 0.31375667  0.47718349]
 [ 0.56689254  0.77079148]
 [ 0.7321604   0.35828963]
 [ 0.15724842  0.94294584]
 [ 0.34933722  0.84634483]
 [ 0.50304053  0.81299619]
 [ 0.23869886  0.9895604 ]
 [ 0.4636501   0.32531094]
 [ 0.36510487  0.97365522]
 [ 0.73350238  0.83833013]
 [ 0.61810158  0.12580353]
 [ 0.59274817  0.18779828]
 [ 0.87150299  0.34679501]
 [ 0.25883219  0.50002932]
 [ 0.75690948  0.83429824]
 [ 0.29316649  0.05646578]
 [ 0.10409134  0.88235166]
 [ 0.06727785  0.57784761]
 [ 0.38492705  0.48384792]
 [ 0.69234428  0.19687348]
 [ 0.42783492  0.73416985]
 [ 0.09696069  0.04883936]]
Y_:
 [[1], [0], [0], [1], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1], [1], [1], [1], [1], [0], [1]]
w1:
 [[-0.81131822  1.48459876  0.06532937]
 [-2.4427042   0.0992484   0.59122431]]
w2:
 [[-0.81131822]
 [ 1.48459876]
 [ 0.06532937]]
After 0 training steps, loss_mse on all data is 5.20999
After 500 training steps, loss_mse on all data is 0.617026
After 1000 training steps, loss_mse on all data is 0.392288
After 1500 training steps, loss_mse on all data is 0.386432
After 2000 training steps, loss_mse on all data is 0.384254
After 2500 training steps, loss_mse on all data is 0.383676
w1:
 [[-0.40074909  1.02251101  1.00135291]
 [-2.13084817 -0.23977895  1.12739885]]
w2:
 [[-0.4457432 ]
 [ 1.04927158]
 [-0.5386759 ]]
'''
