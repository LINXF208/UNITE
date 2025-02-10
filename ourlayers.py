import math
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tensorflow import keras


class GLearnLayer(keras.layers.Layer):
    def __init__(self, hidden_size, num_pers=8, activation=tf.nn.relu):
        """Setting hyperparameters parametermaters of GLearnLayer."""
        super(GLearnLayer,self).__init__()

        self.num_pers = num_pers
        self.hidden_size = hidden_size
        self.activation = activation
        self.loss = 0
        
    def build(self, input_shape):
        """Initilizing parametermaters GLearnLayer.

        Args:
            input_shape: Dimension of input data

        Returns: None
        """

        self.kernel = self.add_variable(
            "kernel", 
            shape = [int(input_shape[-1]), 
            self.hidden_size * self.num_pers]
        )

        self.att_self_weight = self.add_weight(
            name='att_self_weight',
            shape=[1, self.num_pers, self.hidden_size],
            dtype=tf.float32, 
            initializer=tf.keras.initializers.glorot_uniform()
        )
        self.att_neighs_weight = self.add_weight(
            name='att_neighs_weight',
            shape=[1, self.num_pers, self.hidden_size],
            dtype=tf.float32, 
            initializer=tf.keras.initializers.glorot_uniform()
        )

        self.bias_l0 = self.add_weight(
            name='bias', 
            shape=[self.num_pers, int(input_shape[0]), int(input_shape[0])],
            dtype=tf.float32, 
            initializer=keras.initializers.Zeros()
        )

    def call(self, input, training=False):
        """Building interference structure based on sparse-attention.

        Args:
            inputs (tf.Tensor): Input feature matrix.
            training (bool): Flag indicating training mode.

        Returns:
            tf.Tensor: interference structure.
        """
        weighted_input = tf.matmul(input, self.kernel) 
        weighted_input = tf.reshape(weighted_input, [-1, self.num_pers, self.hidden_size])

        attn_for_self = tf.reduce_sum(
            weighted_input * self.att_self_weight,
            axis=-1, 
            keepdims=True
        )  
        attn_for_neighs = tf.reduce_sum(
            weighted_input * self.att_neighs_weight,
            axis=-1,
            keepdims=True
        )

        gamma = tf.cast(-0.1, tf.float32)
        zeta = tf.cast(1.1, tf.float32)
        beta = tf.cast(0.66, tf.float32)
        eps = tf.cast(1e-20, tf.float32)
        const1 = beta * np.log(-gamma / zeta + eps)

        dense = tf.transpose(attn_for_self, [1, 0, 2]) + tf.transpose(attn_for_neighs, [1, 2, 0])
        logits = dense + self.bias_l0

        self.loss = tf.reduce_mean(tf.nn.sigmoid(logits - const1))

        if training:
            U = tf.cast(tf.random.uniform(logits.shape), tf.float32) + eps
            s = tf.nn.sigmoid((tf.math.log(U / (1 - U)) + logits) / beta)
            s_bar = s * (zeta - gamma) + gamma
            dense = tf.clip_by_value(s_bar, 0, 1)

        else:
            s = tf.nn.sigmoid(logits / beta)
            s_bar = s * (zeta - gamma) + gamma
            dense= tf.clip_by_value(s_bar, 0, 1)

        n_eye = tf.eye(dense.shape[-1])
        n_eye = tf.stop_gradient(tf.expand_dims(n_eye, axis=0))
        dense = n_eye+dense*(1 - n_eye) 

        attention = dense / tf.clip_by_value(
            tf.reduce_sum(dense, keepdims=True, axis=-1), 
            1e-10, 
            tf.reduce_sum(dense, keepdims=True, axis=-1)
        )
        attention = tf.reduce_mean(attention, axis = 0)
      
        return attention


class RepLayer(keras.layers.Layer):
    def __init__(self, num_outputs, activation=tf.nn.relu):
        """Initilizing parametermaters RepLayer."""

        super(RepLayer, self).__init__()

        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        """Initilizing parametermaters RepLayer.

        Args:
            input_shape (tuple): Dimension of input data

        Returns: None
        """
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]),
            self.num_outputs],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform()
        )
                                      
        self.bias = self.add_weight(
            "bias",
            shape=[self.num_outputs], 
            initializer=keras.initializers.Zeros()
        )
       
    def call(self, input):
        """Feedforward for RepLayer.

        Args:
            input (tf.Tensor): covaiates

        Returns: Representation of covariates
        """
        output = tf.matmul(input, self.kernel) + self.bias
        output = self.activation(output)        

        return output

class GCNLayer(keras.layers.Layer):
    def __init__(self, out_features, activation=tf.nn.relu):
        """Setting hyperparameters parametermaters of GCNLayer."""
        super(GCNLayer, self).__init__()

        self.out_features = out_features
        self.activation = activation
            
    def build(self, input_shape):
        """Initilizing parametermaters GCNLayer.

        Args:
            input_shape (tuple): Dimension of input data

        Returns: None
        """
        self.kernel = self.add_variable(
            "kernel", 
            shape=[int(input_shape[-1]),
            self.out_features]
        )

        self.bias = self.add_variable("bias", shape=[self.out_features])
        
    def call(self, input, adj):
        """Feedforward for GCNLayer.

        Args:
            input (tf.Tensor): Individual information of all individuals.
            adj (tf.Tensor): Interference structure.

        Returns: Representation of interference.
        """
        support = tf.matmul(input, self.kernel)

        output = tf.matmul(adj, support) + self.bias
        output = self.activation(output)
      
        return output
    
