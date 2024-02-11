from sklearn.utils import shuffle
import tensorflow as tf
import math
import numpy as np
import pandas as pd
# import scipy.stats as stats
import statsmodels.api as sm
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(rc={'figure.figsize':(16,10)}, font_scale=1.3)
import random
  

SQRT_CONST = 1e-10
VERY_SMALL_NUMBER = 1e-10


class GLlayer(keras.layers.Layer):
    def __init__(self,hidden_size,num_pers=8,activation=tf.nn.relu,sigma=1):
        super(GLlayer,self).__init__()
        self.num_pers = num_pers
        self.hidden_size = hidden_size
        self.activation = activation
        self.sigma = sigma
        self.loss = 0
        
    def build(self, input_shape):
       
        self.kernel = self.add_variable("kernel", shape = [int(input_shape[-1]), self.hidden_size * self.num_pers])

        self.att_self_weight = self.add_weight(name='att_self_weight',shape=[1, self.num_pers,self.hidden_size],
            dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
        self.att_neighs_weight = self.add_weight(name='att_neighs_weight',shape=[1, self.num_pers,self.hidden_size],
            dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
        self.bias_l0 = self.add_weight(name='bias', shape=[self.num_pers, int(input_shape[0]), int(input_shape[0])],
            dtype=tf.float32,initializer=keras.initializers.Zeros())
        
    def call(self,input,training=False):
        #print("training",training)

        #print("weighted_input",weighted_input)
        #print("w_input_norm",weighted_input_norm)
        # print("weighted_input_norm  has nan:",np.isnan(weighted_input_norm).any())

        
        weighted_input = tf.matmul(input,self.kernel) 
        weighted_input = tf.reshape(weighted_input, [-1, self.num_pers, self.hidden_size])
        #weighted_input = tf.reshape(weighted_input_norm,perm=[1,0,2])
        #print("weighted_input_norm ",weighted_input_norm.shape)
        attn_for_self = tf.reduce_sum(
            weighted_input * self.att_self_weight, axis=-1, keepdims=True)  # None head_num 1
        #print("att self",attn_for_self.shape)
        attn_for_neighs = tf.reduce_sum(
            weighted_input * self.att_neighs_weight, axis=-1, keepdims=True)
        #print("att ne",attn_for_neighs.shape)
        #print("weighted_input_norm",weighted_input_norm)
        #print()
        gamma = tf.cast(-0.1,tf.float32)
        zeta = tf.cast(1.1,tf.float32)
        beta = tf.cast(0.66,tf.float32)
        eps = tf.cast(1e-20,tf.float32)
        const1 = beta*np.log(-gamma/zeta + eps)

        
        dense = tf.transpose(
            attn_for_self, [1, 0, 2]) + tf.transpose(attn_for_neighs, [1, 2, 0])
        #print(dense.shape)
      
        logits = dense+self.bias_l0
                
         


            #self.loss = tf.reduce_mean(tf.nn.sigmoid(tf.clip_by_value(logits - const1,VERY_SMALL_NUMBER,1-VERY_SMALL_NUMBER)))
            #self.loss = tf.reduce_sum(tf.nn.sigmoid(logits - const1))
        self.loss = tf.reduce_mean(tf.nn.sigmoid(logits - const1))
        #print("gl self loss",self.loss)
        #print("gl loss",self.loss)
        if training:
            #U = tf.cast(tf.random.uniform(logits.shape,minval=VERY_SMALL_NUMBER ,maxval = 1-VERY_SMALL_NUMBER),tf.float32) + eps
            U = tf.cast(tf.random.uniform(logits.shape),tf.float32) + eps

            #print("u",U)
            #s = tf.nn.sigmoid((tf.math.log(tf.clip_by_value( U / (1 - U) ,VERY_SMALL_NUMBER,1- VERY_SMALL_NUMBER)) + logits) / beta)
            s = tf.nn.sigmoid((tf.math.log(U / (1 - U)) + logits) / beta)
            #print("s",s)
            s_bar = s * (zeta - gamma) + gamma
            #print("s_bar",s_bar)
            dense = tf.clip_by_value(s_bar, 0, 1)
        else:
            #s = tf.nn.sigmoid(tf.clip_by_value(logits / beta,VERY_SMALL_NUMBER,1-VERY_SMALL_NUMBER))
            s = tf.nn.sigmoid(logits / beta)
            s_bar = s * (zeta - gamma) + gamma
            dense= tf.clip_by_value(s_bar, 0, 1)

        n_eye = tf.eye(dense.shape[-1])
        n_eye = tf.stop_gradient(tf.expand_dims(n_eye, axis=0))
        dense = n_eye+dense*(1-n_eye) 
        attention = dense/tf.clip_by_value(tf.reduce_sum(dense,keepdims = True,axis=-1),VERY_SMALL_NUMBER,tf.reduce_sum(dense,keepdims = True,axis=-1))
        attention = tf.reduce_mean(attention, axis = 0)
        #print("attetnion",attention)


      
        return attention


class reprelayer(keras.layers.Layer):
    def __init__(self, num_outputs,activation=tf.nn.relu):
        """ representation layer
        input:
              num_outputs:  hidden shape
              activation: the activation function of representation layer
        Output:
              The representation of current layer
        """
        super(reprelayer,self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation
    def build(self,input_shape):
        self.kernel = self.add_weight("kernel",shape = [int(input_shape[-1]),self.num_outputs],
                                      dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
        
                                      
        self.bias = self.add_weight("bias",shape=[self.num_outputs], initializer=keras.initializers.Zeros())
       
    def call(self,input,flag=False):
        output = tf.matmul(input,self.kernel)+self.bias
        output = self.activation(output)        
        self.result = output
        return output

class GCNLayer(keras.layers.Layer):
    def __init__(self,out_features,batch_norm = False,activation = tf.nn.relu):
        super(GCNLayer,self).__init__()
        self.out_features = out_features
        self.activation = activation

        if batch_norm:
            self.batch_norm = batch_norm
        else:
            self.batch_norm = None
            
    def build(self,input_shape):
        self.kernel = self.add_variable("kernel", shape = [int(input_shape[-1]),self.out_features])
        self.bias = self.add_variable("bias",shape=[self.out_features])
        
    def call(self,input,adj):
        support = tf.matmul(input,self.kernel)
        output = tf.matmul(adj,support)+self.bias
        output = self.activation(output)
      
        return output
    
