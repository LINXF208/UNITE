#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.utils import shuffle

import pandas as pd
import math
import utils
import ourlayers
from tensorflow import keras
import random



SQRT_CONST = 1e-10
VERY_SMALL_NUMBER = 1e-10

class UNITE(keras.Model):
    def __init__(self,config,activation=tf.nn.relu):
        super(UNITE, self).__init__()
       

        self.adj = None
        print("Initialization ...")

        #self.norm_adj = utils.normalize_adj(init_adj)

        self.rep_layers = []
        self.gnn_layers = []
        self.out_T_layers = []
        self.out_C_layers = []
        self.out_layers = []
        self.train_loss = None


        self.activation = activation

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
        self.use_batch = config['use_batch']
       
        self.optimizer = keras.optimizers.Adam(lr=config['lr_rate'],decay = config['lr_weigh_decay'])
 

        self.GLlayes = []
        for i in range(len(config['GL_hidden_shape'])):
            curGL = ourlayers.GLlayer(hidden_size = config['GL_hidden_shape'][i], num_pers = config['head_num_gl'] ,
                activation=self.activation)
            self.GLlayes.append(curGL)

        for i in range(config['rep_hidden_layer']):
            h = ourlayers.reprelayer(config['rep_hidden_shape'][i],activation = self.activation)
            self.rep_layers.append(h)
        

        for i in range(config['GNN_hidden_layer']):
            g = ourlayers.GCNLayer(out_features = config['GNN_hidden_shape'][i],activation = self.activation)
            self.gnn_layers.append(g)
           
 
        
    

        
        for i in range(config['out_T_layer']):
            out_T = keras.layers.Dense(config['out_hidden_shape'][i], activation = self.activation,kernel_initializer=tf.keras.initializers.glorot_uniform())

            self.out_T_layers.append(out_T)
   
        for i in range(config['out_C_layer']):
            out_C = keras.layers.Dense(config['out_hidden_shape'][i], activation = self.activation,kernel_initializer=tf.keras.initializers.glorot_uniform())
            #o_hidden_shape = o_hidden_shape//2
            self.out_C_layers.append(out_C)
       
        self.layer_6_T = keras.layers.Dense(1)
        self.layer_6_C = keras.layers.Dense(1)
        self.result = None
            

     
                
            
        
    def call(self, inputtensor,train_idx,training = False):
        #print("CV ...")
        input_tensor = inputtensor[:,:-1]
        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])

        #bias_list = tf.ones([len(input_tensor),1])
        features = input_tensor
        #input_tensor = tf.concat([input_tensor,bias_list],1)
        
        
        regularization = 0
        
        
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
        if self.flag_norm_rep:
            h_rep_norm = hidden /tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(hidden),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else:
            h_rep_norm = hidden * 1.0

        mask_rep_t = tf.concat([h_rep_norm,input_t],axis = 1)
      
        cur_adj = self.GLlayes[0](mask_rep_t,training=False)
        self.adj = cur_adj 

        
        GNN = mask_rep_t
        #if self.gnn_sel == 'gcn':
        for i in range(len(self.gnn_layers)):
              
            GNN = self.gnn_layers[i](GNN,self.adj)
           
       
        if self.flag_norm_gnn:
            GNN_norm = GNN/ tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(GNN),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else: 
            GNN_norm = GNN*1.0


        concated_data = tf.concat([h_rep_norm,GNN_norm],axis = 1)


        
        train_concated_data = tf.gather(concated_data,train_idx)
        train_input_t = tf.gather(input_t,train_idx)
        train_hidden = tf.gather(h_rep_norm,train_idx)
        train_GNN = tf.gather(GNN_norm,train_idx)

       

    
       
        outnn_T = train_concated_data 
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
           
        
        output_T = self.layer_6_T(outnn_T)

        outnn_C = train_concated_data 
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
         
           

        output_C = self.layer_6_C(outnn_C)

       
      

        return output_T,output_C
    
   
    
        



       
            
            
            

    
    def get_loss(self,inputtensor,all_y,train_idx,training = True):
        input_tensor = inputtensor[:,:-1]
        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])

        #bias_list = tf.ones([len(input_tensor),1])
        features = input_tensor
        #input_tensor = tf.concat([input_tensor,bias_list],1)
        
        input_tensor = tf.nn.dropout(input_tensor,self.inp_drop)
        
        regularization = 0
        
        
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
            hidden = tf.nn.dropout(hidden,self.rep_dropout)
            #regularization += tf.nn.l2_loss(self.rep_layers[i].kernel)
        if self.flag_norm_rep:
            h_rep_norm = hidden /tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(hidden),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else:
            h_rep_norm = hidden * 1.0
            
        mask_rep_t = tf.concat([h_rep_norm,input_t],axis = 1)
     
        #print("mask_rep",mask_rep_t)


        #cur_adj = self.GLlayes[0](input_tensor)
        cur_adj = self.GLlayes[0](mask_rep_t,training=True)
        #print("cur_adj sum",tf.reduce_sum(tf.sign(cur_adj)))
        self.adj = tf.nn.dropout(cur_adj,self.GL_dropout)
        regularization += self.reg_lambda*self.GLlayes[0].loss
       
        
        GNN = mask_rep_t
        #if self.gnn_sel == 'gcn':
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN,self.adj)
            GNN = tf.nn.dropout(GNN,self.GNN_dropout)
            #regularization += tf.nn.l2_loss(self.gnn_layers[i].kernel)
        
       
        if self.flag_norm_gnn:
            GNN_norm = GNN/ tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(GNN),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else: 
            GNN_norm = GNN*1.0


        concated_data = tf.concat([h_rep_norm,GNN_norm],axis = 1)


        
        train_concated_data = tf.gather(concated_data,train_idx)

        train_input_t = tf.gather(input_t,train_idx)
        train_y = tf.gather(all_y,train_idx)
        train_hidden = tf.gather(h_rep_norm,train_idx)
        train_GNN = tf.gather(GNN_norm,train_idx)

        if self.use_batch:
            I = random.sample(range(0, len(train_concated_data)), self.use_batch)
            train_concated_data = tf.gather(train_concated_data,I)
            train_input_t = tf.gather(train_input_t,I)
            train_y = tf.gather(train_y ,I)
            train_hidden = tf.gather(train_hidden,I)
            train_GNN = tf.gather(train_GNN ,I)


    
        group_t,group_c,i_0,i_1= utils.divide_TC(train_concated_data ,train_input_t)

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T,self.out_dropout)
            #regularization += tf.nn.l2_loss(self.out_T_layers[i].kernel)

        
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C =  tf.nn.dropout(outnn_C,self.out_dropout)
            #regularization += tf.nn.l2_loss(self.out_C_layers[i].kernel)

           

        output_C = self.layer_6_C(outnn_C)

        y_pre = tf.dynamic_stitch([i_0,i_1],[output_C,output_T])

        p_t = tf.divide(tf.reduce_sum(train_input_t),train_input_t.shape[0])
        
        
        clf_error_1_pri = tf.reduce_mean(tf.square(train_y - y_pre))   
        pred_error_1 = clf_error_1_pri 
        print("Train loss", pred_error_1)

        rep_error_1 = self.rep_alpha * utils.COMP_HSIC(train_hidden,train_input_t)
        #GNN_error_1 = self.GNN_alpha * self.COMP_HSIC(train_GNN,train_input_t)
        cross_error = self.cross_alpha * utils.COMP_HSIC(train_GNN,train_hidden)
        print("hsic rep_loss",rep_error_1,"cross_loss",cross_error)

       
        L_1 =   rep_error_1 + pred_error_1 + cross_error + regularization
        print("total loss",L_1)

        return L_1
    
   
    
    def get_grad(self,inputtensor,y,train_idx):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(inputtensor,y,train_idx)
            self.train_loss = L
            g = tape.gradient(L,self.variables)
        return g
        
    def network_learn(self, inputtensor,y,train_idx):
        g = self.get_grad(inputtensor,y,train_idx)
        self.optimizer.apply_gradients(zip(g,self.variables))
        return self.train_loss
        
 
    
  
            
    def CV_y(self,inputtensor,y,test_idx,training = False):
        print("CV ...")

        input_tensor = inputtensor[:,:-1]
        #print("inp shape",input_tensor.shape)

        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])

        hidden = input_tensor

        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
        #print("rep shape",hidden.shape)
        if self.flag_norm_rep:
            h_rep_norm = hidden /tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(hidden),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else:
            h_rep_norm = hidden*1.0
     

        mask_rep_t = tf.concat([h_rep_norm,input_t],axis = 1)
        
    
   
        cur_adj = self.GLlayes[0](mask_rep_t)    
        self.adj = cur_adj
        GNN = mask_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN,self.adj)

   
        if self.flag_norm_gnn:
            GNN_norm = GNN/ tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(GNN),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else:
            GNN_norm = GNN*1.0
        concated_data = tf.concat([h_rep_norm,GNN_norm],axis = 1)
        train_concated_data = tf.gather(concated_data,test_idx)

        train_input_t = tf.gather(input_t,test_idx)
        train_y = tf.gather(y,test_idx)

        group_t,group_c,i_0,i_1 = utils.divide_TC(train_concated_data ,train_input_t)

        p = tf.divide(tf.reduce_sum(train_input_t),train_input_t.shape[0])
        y_T = tf.gather(train_y,i_1)
        y_C = tf.gather(train_y,i_0)
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            #print("outnn_T shape",outnn_T.shape)


        output_T = self.layer_6_T(outnn_T)
        #print("outnn_T shape",outnn_T.shape)


        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            #print("outnn_C shape",outnn_C.shape)

        #print("outnn_C shape",outnn_C.shape)

        output_C = self.layer_6_C(outnn_C)

        clf_error_T_1_pri = tf.reduce_mean(tf.square(y_T - output_T))
        clf_error_C_1_pri = tf.reduce_mean(tf.square(y_C - output_C))

        pred_error_1 = clf_error_T_1_pri+clf_error_C_1_pri
        print("cv","T loss", clf_error_T_1_pri,"C loss", clf_error_C_1_pri)
                 
        return pred_error_1
    
  


    def pre_no_interf(self,inputtensor,test_idx,only_T = False,training = False):
        

  
        input_tensor = inputtensor[:,:-1]

        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])

        features = input_tensor

    

        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
        if self.flag_norm_rep:
            h_rep_norm = hidden /tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(hidden),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        else:
            h_rep_norm = hidden*1.0
  

        mask_rep_t = tf.concat([h_rep_norm,input_t],axis = 1)
       

        cur_adj = self.GLlayes[0](mask_rep_t)
        self.adj = cur_adj

        GNN = mask_rep_t
        for i in range(len(self.gnn_layers)):
            if self.rebuild_flag:
                cur_adj = self.GLlayes[i](GNN)
                print("c adj",cur_adj)
                print(str(i)+"cur_adj sum",tf.reduce_sum(tf.sign(cur_adj)))
                
                self.adj = cur_adj
            GNN = self.gnn_layers[i](GNN,self.adj)

      
        zeros = tf.zeros_like(GNN)
        concated_data = tf.concat([h_rep_norm,zeros],axis = 1)
        test_input_t = tf.gather(input_t,test_idx)
        
       
      
        train_concated_data = tf.gather(concated_data,test_idx)
        if only_T:
            it = tf.where(test_input_t>0)[:,0]
            test_input_t = tf.gather(test_input_t,it)
            train_concated_data = tf.gather(train_concated_data,it)

        outnn_T = train_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.layer_6_T(outnn_T)

        outnn_C = train_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.layer_6_C(outnn_C)
        
        
        return output_T,output_C 

   