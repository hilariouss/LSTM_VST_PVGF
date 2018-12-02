import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import xavier_initializer


class Model_RNN:
    def __init__(self,input_dim,output_dim,duration,modelname = "",X = None, n_hidden = 128, n_features = 128):
        self.modelname = modelname
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.duration = duration

        self.n_hidden = n_hidden
        self.n_features = n_features

        if(X == None):
            self.X = tf.placeholder(tf.float32,shape=[None,self.duration,self.input_dim]) #None,step,input
        else:
            self.X = X

        self.logits, self.output_features = self.build_model()
        self.logits_sigmoid = tf.nn.sigmoid(self.logits)
        self.logits_relu = tf.nn.relu(self.logits)



    def build_model(self):
        '''
        with tf.variable_scope(self.modelname+"layer") as scope:
            self.Wf = tf.get_variable(name ='weights_f',shape=[self.n_hidden,self.n_features], initializer=xavier_initializer())
            self.Bf = tf.get_variable(name ='weights_b_f',shape=[self.n_features], initializer=xavier_initializer())


            self.Wo = tf.get_variable(name ='weights_o',shape=[self.n_features,self.output_dim], initializer=xavier_initializer())
            self.Bo = tf.get_variable(name ='weights_b_o',shape=[self.output_dim], initializer=xavier_initializer())
       '''

        with tf.variable_scope(self.modelname+"rnncell") as scope:
            cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden,name=self.modelname+"cell1")
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1,output_keep_prob=0.7)
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden,name=self.modelname+"cell2")

        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1,cell2],)
        outputs,states = tf.nn.dynamic_rnn(multi_cell,self.X,dtype=tf.float32)
        outputs = tf.transpose(outputs,[1,0,2])
        outputs = outputs[-1]

        output_features = tf.layers.dense(outputs,units=self.n_features,activation = tf.nn.relu)
        outputs = tf.layers.dense(output_features,units=self.n_features,activation = None)
        outputs = tf.layers.dense(outputs,units=self.output_dim,activation = None)
        return outputs, output_features



def modelTester():
    Model_RNN(5,1,6)
    pass

if __name__ == "__main__":
    modelTester()
    pass
