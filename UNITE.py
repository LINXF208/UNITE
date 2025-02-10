import math
import random

import numpy as np 
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import utils
import ourlayers


class UNITE(keras.Model):
    def __init__(self, config, activation=tf.nn.relu):
        super(UNITE, self).__init__()
        """Initilize UNITE model."""
        print("Initialization ...")

        self.rep_layers = []
        self.gnn_layers = []
        self.out_T_layers = []
        self.out_C_layers = []
        self.GL_layers = []

        self.train_loss = None
        self.adj = None
        self.activation = activation
        self.use_batch = config['use_batch']
        self.optimizer = keras.optimizers.Adam(lr=config['lr_rate'], decay = config['lr_weigh_decay'])

        self.cross_alpha = config['cross_alpha']
        self.rep_alpha = config['rep_alpha']
        self.reg_lambda = config['reg_lambda']

        self.flag_norm_gnn= config['flag_norm_gnn']
        self.flag_norm_rep= config['flag_norm_rep']

        self.out_dropout = config['out_dropout']
        self.GNN_dropout = config['GNN_dropout']
        self.rep_dropout = config['rep_dropout']
        self.inp_drop = config['inp_dropout']
        self.GL_dropout = config['GL_dropout']

        for i in range(len(config['GL_hidden_shape'])):
            gl = ourlayers.GLearnLayer(
                hidden_size=config['GL_hidden_shape'][i],
                 num_pers=config['head_num_gl'],
                activation=self.activation
            )
            self.GL_layers.append(gl)

        for i in range(config['rep_hidden_layer']):
            h = ourlayers.RepLayer(config['rep_hidden_shape'][i], activation=self.activation)
            self.rep_layers.append(h)

        for i in range(config['GNN_hidden_layer']):
            g = ourlayers.GCNLayer(out_features=config['GNN_hidden_shape'][i], activation=self.activation)
            self.gnn_layers.append(g)

        for i in range(config['out_T_layer']):
            out_t = keras.layers.Dense(config['out_hidden_shape'][i], 
                activation=self.activation, 
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            )
            self.out_T_layers.append(out_t)
   
        for i in range(config['out_C_layer']):
            out_c = keras.layers.Dense(config['out_hidden_shape'][i], 
                activation=self.activation, 
                kernel_initializer=tf.keras.initializers.glorot_uniform()
            )
            #o_hidden_shape = o_hidden_shape//2
            self.out_C_layers.append(out_c)
       
        self.final_out_y1 = keras.layers.Dense(1)
        self.final_out_y0 = keras.layers.Dense(1)

    def call(self, input_tensor, train_idx, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0

        concated_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        self.adj = self.GL_layers[0](concated_rep_t, training=False)

        GNN = concated_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN, self.adj)
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else: 
            GNN_norm = GNN * 1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm], axis=1)

        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_hidden = tf.gather(h_rep_norm, train_idx)
        train_GNN = tf.gather(GNN_norm, train_idx)

        outnn_T = train_concated_data 
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = train_concated_data 
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)

        return output_T, output_C

    def get_loss(self, input_tensor, all_y, train_idx, training=True):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])
        input_x = tf.nn.dropout(input_x, self.inp_drop)

        regularization = 0

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
            hidden = tf.nn.dropout(hidden, self.rep_dropout)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0

        concated_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        self.adj = self.GL_layers[0](concated_rep_t, training=True)
        self.adj = tf.nn.dropout(adj, self.GL_dropout)
        regularization += self.reg_lambda * self.GL_layers[0].loss

        GNN = concated_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN, self.adj)
            GNN = tf.nn.dropout(GNN, self.GNN_dropout)
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else: 
            GNN_norm = GNN * 1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm], axis=1)

        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(all_y, train_idx)
        train_hidden = tf.gather(h_rep_norm, train_idx)
        train_GNN = tf.gather(GNN_norm, train_idx)

        if self.use_batch:
            indices = random.sample(range(0, len(train_concated_data)), self.use_batch)
            train_concated_data = tf.gather(train_concated_data, indices)
            train_input_t = tf.gather(train_input_t, indices)
            train_y = tf.gather(train_y, indices)
            train_hidden = tf.gather(train_hidden, indices)
            train_GNN = tf.gather(train_GNN, indices)

        group_t, group_c, i_0, i_1= utils.divide_t_c(train_concated_data, train_input_t)

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T, self.out_dropout)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C =  tf.nn.dropout(outnn_C, self.out_dropout)
        output_C = self.final_out_y0(outnn_C)

        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])
        p_t = tf.divide(tf.reduce_sum(train_input_t), train_input_t.shape[0])

        pred_error = tf.reduce_mean(tf.square(train_y - y_pre))   
        print("Train loss", pred_error)

        rep_error_1 = self.rep_alpha * utils.comp_hsic(train_hidden, train_input_t)
        cross_error = self.cross_alpha * utils.comp_hsic(train_GNN, train_hidden)
        print("hsic rep_loss", rep_error_1, "cross_loss", cross_error)

        L_1 =   rep_error_1 + pred_error + cross_error + regularization
        print("total loss", L_1)

        return L_1

    def get_grad(self, input_tensor, y, train_idx):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            self.train_loss = self.get_loss(input_tensor, y, train_idx)
            g = tape.gradient(self.train_loss, self.variables)

        return g

    def network_learn(self, input_tensor, y, train_idx):
        g = self.get_grad(input_tensor, y, train_idx)
        self.optimizer.apply_gradients(zip(g, self.variables))

        return self.train_loss

    def val_y(self, input_tensor, y, val_idx, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])

        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0

        concated_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        self.adj = self.GL_layers[0](concated_rep_t)

        GNN = concated_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN, self.adj)
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            GNN_norm = GNN * 1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm], axis=1)

        val_concated_data = tf.gather(concated_data, val_idx)
        val_input_t = tf.gather(input_t, val_idx)
        val_y = tf.gather(y, val_idx)
        group_t, group_c, i_0, i_1 = utils.divide_t_c(val_concated_data, val_input_t)

        p = tf.divide(tf.reduce_sum(val_input_t), val_input_t.shape[0])

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)

        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])

        pred_error = tf.reduce_mean(tf.square(val_y - y_pre))     

        return pred_error

    def pre_no_interf(self, input_tensor, test_idx, only_t=False, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis = 1, keepdims=True), 1e-10 np.inf))
        else:
            h_rep_norm = hidden*1.0

        concated_rep_t = tf.concat([h_rep_norm, input_t],axis = 1)

        self.adj = self.GL_layers[0](concated_rep_t)

        GNN = concated_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN, self.adj)

        zeros = tf.zeros_like(GNN)
        concated_data = tf.concat([h_rep_norm, zeros], axis = 1)

        test_input_t = tf.gather(input_t, test_idx)
        train_concated_data = tf.gather(concated_data, test_idx)
        if only_t:
            it = tf.where(test_input_t > 0)[:, 0]
            test_input_t = tf.gather(test_input_t, it)
            train_concated_data = tf.gather(train_concated_data, it)

        outnn_T = train_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = train_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)

        return output_T, output_C 

   