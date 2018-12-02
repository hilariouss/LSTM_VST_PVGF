import tensorflow as tf
import numpy as np
from dataset_loader import Dataset_loader
from model_rnn_regression import Model_RNN
from model_cnn_regression3 import Model_CNN
from tensorflow.contrib.layers import xavier_initializer
from tqdm import tqdm

class Trainer:
    def __init__(self):
        #parameters
        self.totalEpoch = 8000
        self.batchSize = 128
        self.batchSize_test = 128


        #dataset
        self.dataset_loader = Dataset_loader(pvdir = "./data/pv_2015_2016_gy_processed.csv",duration_hour =6,duration_hour_long=24*7)
        self.trainset,self.testset = self.dataset_loader.getDataset(shuffle = True)


        print(len(self.trainset))
        print(len(self.testset))

        #model tensor :
        self.numClasses = 1
        self.cnn = Model_CNN(input_dim = self.dataset_loader.num_attribute,output_dim = self.numClasses,duration=self.dataset_loader.duration,modelname="shortRNN")
        self.rnn_long = Model_RNN(input_dim = self.dataset_loader.num_attribute,output_dim = self.numClasses,duration=self.dataset_loader.duration_long,modelname="longRNN")
        self.concated_features = tf.concat([self.cnn.output_features,self.rnn_long.output_features],1)
        print(self.concated_features)

        self.W = tf.get_variable(name ='weights',shape=[self.concated_features.shape[1],1], initializer=xavier_initializer())
        self.B = tf.get_variable(name='weights_b', shape=[1], initializer=xavier_initializer())

        self.model_logits = tf.matmul(self.concated_features,self.W)*self.B
        self.Y = tf.placeholder(tf.float32, shape=[None,1])




    def train(self):
    # ===============================================
    # 1. declare writers for tensorboard
    # ===============================================




    # ==========================================================
    # 2. define cost function and train step
    # ==========================================================
        with tf.name_scope("l2_loss") as scope:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * 0.0005

        with tf.name_scope("tendancy_loss") as scope:
            y_t = tf.strided_slice(self.Y,[0,0],[self.batchSize-1,1],strides =[2,1]) - tf.strided_slice(self.Y,[1,0],[self.batchSize,1],strides=[2,1])
            ypred_t = tf.strided_slice(self.model_logits, [0, 0], [self.batchSize - 1, 1],strides=[2,1]) - tf.strided_slice(self.model_logits, [1, 0],
                                                                               [self.batchSize, 1],strides=[2,1])
            tlossrate = 0
            tloss = tf.reduce_mean(tf.abs(y_t-ypred_t))

        with tf.name_scope("trainer") as scope:
            loss = tf.reduce_mean(tf.losses.mean_squared_error(labels= self.Y,predictions=self.model_logits))
            #loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Y- self.model.logits_relu),reduction_indices=1))+lossL2
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss + tlossrate * tloss)  # +lossL2)

        with tf.name_scope("evaluation") as scope:
            correct_prediction = tf.squeeze(tf.reduce_mean(
                (tf.abs((self.Y) - (self.model_logits)) / (self.Y + self.model_logits)),
                reduction_indices=0)*200)  # smape
            print(correct_prediction)
            correct_prediction_square = tf.reduce_mean(tf.square((self.Y) - (self.model_logits)), reduction_indices=1)

            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # ===============================================
    # 3. train
    # ===============================================
    # ===================== main =======================
    # Option set config
    # ==================================================
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #===============================================
        # 3.1 session init and graph load
        #===============================================
            sess.run(tf.global_variables_initializer())

        # ===============================================
        # 1. declare writers for tensorboard
        # ==============================================
            accuracy_hist = tf.placeholder(tf.float32)
            accuracy_hist_summary = tf.summary.scalar('acc_hist', accuracy_hist)

            accuracy_s_hist = tf.placeholder(tf.float32)
            accuracy_s_hist_summary = tf.summary.scalar('acc_square_hist', accuracy_s_hist)

            loss_hist_summary = tf.summary.scalar('training_loss_hist', loss)
            merged = tf.summary.merge_all()
            writer_acc_loss = tf.summary.FileWriter("./board_cnn_rnn_long/acc_loss", sess.graph)

            prediction_hist = tf.placeholder(tf.float32)
            prediction_hist_summary = tf.summary.scalar('pred_hist', prediction_hist)
            prediction_hist_merged = tf.summary.merge([prediction_hist_summary])

            writer_pred = tf.summary.FileWriter("./board_cnn_rnn_long/pred", sess.graph)
            writer_pred_label = tf.summary.FileWriter("./board_cnn_rnn_long/pred_label", sess.graph)


        #===============================================
        # 3.2. train loop
        #===============================================
            for e in range(self.totalEpoch):
            # ...........................
            # shuffle trainset
            # ...........................
                '''
                p = np.random.permutation(int(len(self.trainset) / self.batchSize))
                p2 = []
                for p_ in p:
                    for i in range(self.batchSize):
                        p2.append(p_ * self.batchSize + i)
                self.trainset = np.array(self.trainset)[p2]
                '''
                shuffle_chunksize = 1
                p = np.random.permutation(int(len(self.trainset) / shuffle_chunksize))
                p2 = []
                for p_ in p:
                    for i in range(shuffle_chunksize):
                        p2.append(p_ * shuffle_chunksize + i)
                self.trainset = np.array(self.trainset)[p2]
            # ..........................
            # 3.2.1 학습
            # .........................

                loss_list = []
                tloss_list = []
                train_accuracy_list = []
                for i in tqdm(range(int(len(self.trainset) / self.batchSize))):
                    # == batch load
                    batch_x = np.array([self.trainset[b].get2DShapeInput()  for b in range(i*self.batchSize,(i+1)*self.batchSize)])
                    batch_x_long = np.array([np.squeeze(self.trainset[b].get2DShapeInput_long(),axis=-1)  for b in range(i*self.batchSize,(i+1)*self.batchSize)])

                    batch_y = np.array([self.trainset[b].pv_label  for b in range(i*self.batchSize,(i+1)*self.batchSize)]).reshape(self.batchSize,-1)

                    # == train
                    loss_print, tloss_print,_ = sess.run([loss,tloss,train_step], feed_dict={self.cnn.X: batch_x,self.rnn_long.X: batch_x_long, self.Y: batch_y, self.cnn.trainphase: True})


                    # == logging

                    loss_list.append(loss_print)
                    tloss_list.append(tloss_print)
                print("반복(Epoch):", e, "트레이닝 데이터 정확도:", np.mean(train_accuracy_list), "손실 함수(loss):",
                      np.mean(loss_list),"tloss:",np.mean(tloss_list),"trate",tlossrate)

            # ..........................
            # 3.2.2 평가
            # .........................
            # 매epoch 마다 test 데이터셋에 대한 정확도와 loss를 출력.
                test_accuracy_list = []
                test_accuracy_s_list = []

                loss_list = []
                tloss_list = []
                histloginterval = 100
                if (e % histloginterval == 0):
                    writer_pred = tf.summary.FileWriter("./board_cnn_rnn_long/pred" + str(e), sess.graph)
                    writer_pred_label = tf.summary.FileWriter("./board_cnn_rnn_long/pred_label" + str(e), sess.graph)

                for i in range(int(len(self.testset) / self.batchSize_test)):
                    # == test batch load
                    test_batch_x = np.array([self.testset[b].get2DShapeInput()  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)])
                    test_batch_x_long = np.array([np.squeeze(self.testset[b].get2DShapeInput_long(),axis=-1)  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)])

                    test_batch_y = np.array([self.testset[b].pv_label  for b in range(i*self.batchSize_test,(i+1)*self.batchSize_test)]).reshape(self.batchSize_test,-1)

                    # == logging
                    test_accuracy, test_accuracy_square = sess.run([correct_prediction, correct_prediction_square],
                                                                   feed_dict={self.cnn.X: test_batch_x,self.rnn_long.X: test_batch_x_long, self.Y: test_batch_y, self.cnn.trainphase: False})
                    test_accuracy_list.append(test_accuracy)
                    test_accuracy_s_list.append(test_accuracy_square)

                    loss_print, tloss_print= sess.run([loss, tloss],
                                                      feed_dict={self.cnn.X: test_batch_x, self.rnn_long.X: test_batch_x_long,
                                                                 self.Y: test_batch_y, self.cnn.trainphase: False})
                    loss_list.append(loss_print)
                    tloss_list.append(tloss_print)

                    '''
                    if (e % histloginterval == 0):
                        for o in range(len(test_batch_y)):
                            output = self.model_logits.eval(
                                feed_dict={self.cnn.X: test_batch_x, self.rnn_long.X: test_batch_x_long,
                                           self.Y: test_batch_y, self.cnn.trainphase: False})
                            output_scalar = (output[o])
                            #print("정답: ", (test_batch_y[o]), "출력: ", output_scalar, "step: ",       o + i * self.batchSize_test)
                            writer_pred.add_summary(
                                prediction_hist_merged.eval(feed_dict={prediction_hist: float(output_scalar)}),
                                global_step=o + i * self.batchSize_test)
                            # writer_pred.flush()
                            writer_pred_label.add_summary(
                                prediction_hist_merged.eval(feed_dict={prediction_hist: float(test_batch_y[o])}),
                                global_step=o + i * self.batchSize_test)
                            # writer_pred_label.flush()

                summary = merged.eval(
                    feed_dict={self.cnn.X: test_batch_x,self.rnn_long.X:test_batch_x_long, self.Y: test_batch_y, accuracy_hist: np.mean(test_accuracy_list),
                               accuracy_s_hist: np.mean(test_accuracy_s_list), self.cnn.trainphase: False})
                writer_acc_loss.add_summary(summary, global_step=e)
                writer_acc_loss.flush()
                '''
                print("테스트 데이터 정확도:", np.mean(test_accuracy_list), "손실 함수(loss):", np.mean(loss_list),"tloss:", np.mean(tloss_list))
                print()


def test():
    t = Trainer()
    t.train()
if __name__ == "__main__":
    test()
    pass
