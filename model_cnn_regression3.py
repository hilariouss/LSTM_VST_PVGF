import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import xavier_initializer


class Model_CNN:
    def __init__(self,input_dim,output_dim,duration,modelname = ""):
        self.modelname = modelname
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.duration = int(duration)

        self.trainphase = tf.placeholder(tf.bool)

        self.X = tf.placeholder(tf.float32,shape=[None,self.duration,5,1])

        self.logits = self.apply_logit()
        self.logits_sigmoid = tf.nn.sigmoid(self.logits)
        self.logits_relu = tf.nn.relu(self.logits)


    '''
    def linear(self,X,in_dim ,out_dim,name,active_f = tf.nn.relu):
        with tf.variable_scope(self.modelname+name) as scope:
            W = tf.get_variable(name ='weights',shape=[in_dim,out_dim], initializer=xavier_initializer())
            if(active_f != None):
                return active_f(tf.layers.batch_normalization(tf.matmul(X,W),training=self.trainphase))
            else:
                return (tf.layers.batch_normalization(tf.matmul(X,W),training=self.trainphase))

    '''
    '''
    def linear(self, X, in_dim, out_dim, name, active_f=tf.nn.relu):
        with tf.variable_scope(self.modelname + name) as scope:
            W = tf.get_variable(name='weights', shape=[in_dim, out_dim], initializer=xavier_initializer())
            B = tf.Variable(tf.zeros(out_dim))
            if (active_f != None):
                return active_f(tf.matmul(X, W)+B)
            else:
                return (tf.matmul(X, W)+B)

    '''

    def linear(self,X,in_dim ,out_dim,name,active_f = tf.nn.leaky_relu):
        with tf.variable_scope(self.modelname+name) as scope:
            W = tf.get_variable(name ='weights',shape=[in_dim,out_dim], initializer=xavier_initializer())
            X = tf.layers.batch_normalization(X,training=self.trainphase )
            B = tf.Variable(tf.zeros(out_dim))
            if(active_f != None):
                return active_f(tf.matmul(X,W)+B)
            else:
                return tf.matmul(X,W)+B

    def apply_logit(self):

        with tf.variable_scope(self.modelname+"conv_1") as scope:
            self.W_conv1 = tf.get_variable(name='weights', shape=[1, 5, 1, 5],initializer=xavier_initializer_conv2d())
            self.h_bn1 = tf.layers.batch_normalization(tf.nn.conv2d(self.X, self.W_conv1, strides=[1, 1, 1, 1], padding='VALID'),training=self.trainphase)
            self.h_conv1 = tf.nn.leaky_relu(self.h_bn1)

        with tf.variable_scope(self.modelname+"conv_2") as scope:
            self.W_conv2 = tf.get_variable(name='weights', shape=[2, 5, 1, 5],initializer=xavier_initializer_conv2d())
            self.h_bn2 = tf.layers.batch_normalization(tf.nn.conv2d(self.X, self.W_conv2, strides=[1, 1, 1, 1], padding='VALID'),training=self.trainphase)
            self.h_conv2 = tf.nn.leaky_relu(self.h_bn2)

        with tf.variable_scope(self.modelname + "conv_3") as scope:
            self.W_conv3 = tf.get_variable(name='weights', shape=[3, 5, 1, 5], initializer=xavier_initializer_conv2d())
            self.h_bn3 = tf.layers.batch_normalization(tf.nn.conv2d(self.X, self.W_conv3, strides=[1, 1, 1, 1], padding='VALID'), training=self.trainphase)
            self.h_conv3 = tf.nn.leaky_relu(self.h_bn3)

        with tf.variable_scope(self.modelname + "conv_4") as scope:
            self.W_conv4 = tf.get_variable(name='weights', shape=[4, 5, 1, 5],
                                           initializer=xavier_initializer_conv2d())
            self.h_bn4 = tf.layers.batch_normalization(
                tf.nn.conv2d(self.X, self.W_conv4, strides=[1, 1, 1, 1], padding='VALID'), training=self.trainphase)
            self.h_conv4 = tf.nn.leaky_relu(self.h_bn4)


        self.pooled1 = tf.reshape(self.h_conv1,shape=[-1,int(self.duration),5])
        self.pooled1 = tf.reduce_mean(self.pooled1,reduction_indices=2)
        self.pooled1 = tf.squeeze(self.pooled1)


        self.pooled2 = tf.reshape(self.h_conv2,shape=[-1,int(self.duration-1),5])
        self.pooled2 = tf.reduce_mean(self.pooled2,reduction_indices=2)
        self.pooled2 = tf.squeeze(self.pooled2)

        self.pooled3 = tf.reshape(self.h_conv3, shape=[-1, int(self.duration - 2), 5])
        self.pooled3 = tf.reduce_mean(self.pooled3, reduction_indices=2)
        self.pooled3 = tf.squeeze(self.pooled3)
        '''
        self.pooled4 = tf.reshape(self.h_conv4, shape=[-1, int(self.duration - 3), 5])
        self.pooled4 = tf.reduce_mean(self.pooled4, reduction_indices=2)
        self.pooled4 = tf.squeeze(self.pooled4)
        '''
        '''
        self.pooled1_flatten = tf.reshape(self.h_conv1,shape=[-1,int(self.duration)*5])
        self.pooled2_flatten = tf.reshape(self.h_conv2,shape=[-1,int(self.duration-1)*5])
        self.pooled3_flatten = tf.reshape(self.h_conv3, shape=[-1, int(self.duration-2)*5])
        self.pooled4_flatten = tf.reshape(self.h_conv4, shape=[-1, int(self.duration-3)*5])
        '''


        #self.pooled1_flatten = tf.reshape(self.pooled1,shape=[-1,self.duration])
        self.pooled2_flatten = tf.reshape(self.pooled2,shape=[-1,self.duration-1])
        self.pooled3_flatten = tf.reshape(self.pooled3, shape=[-1, self.duration - 2])
        #self.pooled4_flatten = tf.reshape(self.pooled4, shape=[-1, self.duration - 3])

        #self.flatten = tf.concat([self.pooled1_flatten,self.pooled2_flatten,self.pooled3_flatten,self.pooled4_flatten],1)
        self.flatten = tf.concat([self.pooled2_flatten,self.pooled3_flatten],1)

        num_unit = 300
        H = self.linear(self.flatten,self.flatten.shape[1],num_unit,'L1')
        H = self.linear(H,num_unit,num_unit,"L2")
        #H = self.linear(H,num_unit,num_unit,"L3")
        #H = self.linear(H, num_unit, num_unit, "L4")
        #H = self.linear(H, num_unit, num_unit, "L5")
        #H = self.linear(H, num_unit, num_unit, "L6")

        #H = self.linear(H,num_unit,num_unit,"L7")
        #H = self.linear(H,num_unit,num_unit,"L8")
        #H = self.linear(H,num_unit,num_unit,"L9")
        #H = self.linear(H,num_unit,num_unit,"L10")
        #H = self.linear(H,100,100,"L7")
        #H = self.linear(H,100,100,"L8")
        #H = self.linear(H,100,100,"L9")

        self.output_features = self.linear(H,num_unit,128,"Llast")
        self.output = self.linear(self.output_features,128,1,"output",active_f=None)
        return self.output

def modelTester():
    pass

if __name__ == "__main__":
    modelTester()
    pass


'''
    import tensorflow as tf
    import numpy as np
    from keras.datasets.cifar10 import load_data
    from tensorflow.contrib.layers import xavier_initializer_conv2d
    from tensorflow.contrib.layers import xavier_initializer

    if __name__ == "__main__":
        # ==========================================================
        # 1. CIFAR-10 데이터 다운로드 및 데이터 로드
        # ==========================================================
        (x_train, y_train), (x_test, y_test) = load_data()

        # one hot encoding
        y_train_onehot = np.eye(10)[y_train]
        y_test_onehot = np.eye(10)[y_test]

        y_train_onehot = np.squeeze(y_train_onehot)  # (50000,1,10) -> (50000,10)
        y_test_onehot = np.squeeze(y_test_onehot)  # (10000,1,10) -> (10000,10)

        # ==========================================================
        # 2. 인풋, 레이블을 입력받기 위한 플레이스홀더 정의
        # ==========================================================
        x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        y = tf.placeholder(tf.float32, shape=[None, 10])

        # batch norm 을위한 train pahse 정의
        trainphase = tf.placeholder(tf.bool)
        # dropout 을 위한 keepprob 정의
        keepprob = tf.placeholder(tf.float32)

        # ==========================================================
        # 3. 모델 정의
        # ==========================================================
        x_image = x

        def conv(X, in_ch, out_ch, name, trainpahse):
            with tf.variable_scope(name) as scope:
                W_conv = tf.get_variable(name='weights', shape=[3, 3, in_ch, out_ch],
                                         initializer=xavier_initializer_conv2d())
                h_bn = tf.layers.batch_normalization(tf.nn.conv2d(X, W_conv, strides=[1, 1, 1, 1], padding='SAME'),
                                                     training=trainpahse)
                h_conv = tf.nn.relu(h_bn)
            return h_conv

        with tf.variable_scope("block1") as scope:
            h_conv1 = conv(x_image, 3, 64, "Conv1")
            h_conv2 = conv(h_conv1, 64, 64, "Conv2")
            h_conv2_pool = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("block2") as scope:
            h_conv3 = conv(h_conv2_pool, 64, 128, "Conv3")
            h_conv4 = conv(h_conv3, 128, 128, "Conv4")
            h_conv4_pool = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("block3") as scope:
            h_conv5 = conv(h_conv4_pool, 128, 256, "Conv5")
            h_conv6 = conv(h_conv5, 256, 256, "Conv6")
            h_conv6_pool = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("block4") as scope:
            h_conv7 = conv(h_conv6_pool, 256, 512, "Conv7")
            h_conv8 = conv(h_conv7, 512, 512, "Conv8")
            h_conv8_pool = tf.nn.max_pool(h_conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope("block5") as scope:
            h_conv9 = conv(h_conv8_pool, 512, 512, "Conv9")
            h_conv10 = conv(h_conv9, 512, 512, "Conv10")
            h_conv10_pool = tf.nn.max_pool(h_conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # fc layer
        with tf.variable_scope("fclayer") as scope:
            h_conv10_pool_flat = tf.reshape(h_conv10_pool, [-1, 1 * 1 * 512])
            h_conv10_pool_flat_dropout = tf.nn.dropout(h_conv10_pool_flat, keep_prob=keepprob)
            W_fc = tf.get_variable(name='weights', shape=[512, 10], initializer=xavier_initializer())
            b_fc = tf.Variable(tf.constant(0.1, shape=[10]))
            logits = tf.matmul(h_conv10_pool_flat_dropout, W_fc) + b_fc
            y_pred = tf.nn.softmax(logits)



            # ==========================================================
            # 4. 비용함수 정의
            # ==========================================================
        with tf.name_scope("l2_loss") as scope:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0005

        with tf.name_scope("trainer") as scope:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)) + lossL2
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)




            # ==========================================================
            # 5. 학습
            # ==========================================================
        with tf.Session() as sess:
            # ----------------------------------
            # 5.1 세션, 변수 초기화
            # ----------------------------------
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./board/sample", sess.graph)

            # == 정확도를 계산하는 연산.
            with tf.name_scope("evaluation") as scope:
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # ---------------------------------
                # 5.2 Training loop
                # ---------------------------------
            total_epoch = 300
            for e in range(total_epoch):
                # ..........................
                # 5.2.1 학습
                # .........................
                total_size = x_train.shape[0]
                batch_size = 128

                loss_list = []
                train_accuracy_list = []
                for i in range(int(total_size / batch_size)):
                    # == batch load
                    batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                    batch_y = y_train_onehot[i * batch_size:(i + 1) * batch_size]

                    # == train
                    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, trainphase: True, keepprob: 0.7})

                    # == logging
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, trainphase: False, keepprob: 1})
                    loss_print = loss.eval(feed_dict={x: batch_x, y: batch_y, trainphase: False, keepprob: 1})
                    train_accuracy_list.append(train_accuracy)
                    loss_list.append(loss_print)

                print("반복(Epoch):", e, "트레이닝 데이터 정확도:", np.mean(train_accuracy_list), "손실 함수(loss):",
                      np.mean(loss_list))

                # ..........................
                # 5.2.2 평가
                # .........................
                # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
                test_total_size = x_test.shape[0]
                test_batch_size = 128

                test_accuracy_list = []
                for i in range(int(test_total_size / test_batch_size)):
                    # == test batch load
                    test_batch_x = x_test[i * test_batch_size:(i + 1) * test_batch_size]
                    test_batch_y = y_test_onehot[i * test_batch_size:(i + 1) * test_batch_size]

                    # == logging
                    test_accuracy = accuracy.eval(
                        feed_dict={x: test_batch_x, y: test_batch_y, trainphase: False, keepprob: 1})
                    test_accuracy_list.append(test_accuracy)
                print("테스트 데이터 정확도:", np.mean(test_accuracy_list))
                print()
'''