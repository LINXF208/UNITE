import numpy as np 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import math
import os
import evaluation
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io as scio





def COMP_HSIC(X,t,s_x=1,s_y=1):

    """ Computes the HSIC(X,t)"""
    K = GaussianKernelMatrix(X,s_x)
    L = GaussianKernelMatrix(t,s_y)
    m = X.shape[0]
    H = tf.raw_ops.MatrixDiag(diagonal = tf.ones(shape=[m,])) - 1/m
    LH = tf.matmul(L,H)
    HLH = tf.matmul(H,LH)
    KHLH = tf.matmul(K,HLH)
    #print("check hsic",K,KHLH,tf.linalg.trace(KHLH))
    HSIC = tf.linalg.trace(KHLH)/((m-1)**2)
    #print("check hsic",tf.linalg.trace(KHLH))
    #print(KHLH)
    return HSIC

def GaussianKernelMatrix(x,sigma = 1):

    """ Computes the Gaussian Kernel Matrix"""
    pairwise_distances = pdist2sq(x,x) # Computes the squared Euclidean distance
    return tf.exp(-pairwise_distances/2*sigma)

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keepdims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D    

def divide_TC(concated_data,input_t):
    #temp = tf.concat([hidden,GNN,weighted_G],1)
    #print("input_t",input_t)
    i0 = tf.cast((tf.where(input_t < 1)[:,0]),tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:,0]),tf.int32)
    #mask = np.logical_and(np.array(input_t)[:,-1] == 1,1)
    #print("concated_data",concated_data)
    #print("mask",mask)
    group_T = tf.gather(concated_data,i1)
    #print("group_T",group_T)
    group_C = tf.gather(concated_data,i0)


    return tf.constant(group_T),tf.constant(group_C),i0,i1

def split_train_val_test(data,train_ratio,val_ratio,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]

    return train_indices,val_indices,test_indices

def train(Model_name,input_data,y,train_idx,val_idx,config_hyperparameters, max_iterations,flag_early_stop=False,activation = tf.nn.relu):
    cur_all_features = input_data[:,:-1]
    cur_model = Model_name(config_hyperparameters,activation=activation) 
    count = 0
    losslist_CV = []
    sum_loss = 0
    sum_CV_loss = 0
    losslist = []
    for i in range(max_iterations):
        #print("iter",i)
        loss = cur_model.CV_y(tf.cast(input_data,tf.float32),tf.cast(y,tf.float32),train_idx)
        total_loss = cur_model.network_learn(tf.cast(input_data,tf.float32),tf.cast(y,tf.float32),train_idx)
        CV_loss = cur_model.CV_y(tf.cast(input_data,tf.float32),  tf.cast(y,tf.float32),val_idx)
        sum_loss += loss
        sum_CV_loss += CV_loss
        if (i+1) % 20 == 0:
            if len(losslist_CV) > 0 and sum_CV_loss/20 >= losslist_CV[-1]:
                count += 1
            else:
                count = 0
            if flag_early_stop:
                if i > 400 and count >= 1:
                    break
            losslist.append(sum_loss/20)
            losslist_CV.append(sum_CV_loss/20)
            sum_loss = 0
            sum_CV_loss = 0

    return cur_model

 
def save_mymodel(save_path,save_name,need_save_model):
	cur_path = save_path + '/' + save_name
	need_save_model.save_weights(cur_path )
	print("Already saved the model's weights in file" + cur_path  )

def load_mymodel(load_path,load_name,need_load_model,config_hyperparameters,activation,init_A):
	cur_model = need_load_model(config_hyperparameters,activation,init_A)
	cur_path = load_path + '/' + load_name
	cur_model.load_weights(cur_path)
	print("load model")
	return cur_model




def implement_UNITE(config,data_name,Model_name,activation,cur_i):
   

    if data_name == 'AMZ_neg':

        load_datas = pd.read_csv("./data/AmazonItmFeatures_neg.csv",header=None,prefix="col")
        data = load_datas.to_numpy()
        #prod_G = np.load("./data/new_product_graph_neg.npz")
        #build A obs
        #datas = prod_G['data']
        #indices = prod_G['indices']
        #indptr = prod_G['indptr']
        #shape = prod_G['shape']
        #csr_mat = csr_matrix((datas, indices, indptr), dtype=int)
        #arr = csr_mat.toarray()
        #A = arr[:14538,:14538]
        
        # data prepare
        T = data[:,0]
        y1 = data[:,1]
        y0 = data[:,2]
        T = T.reshape(len(T),1)
        y1 = y1.reshape(len(y1),1)
        y0 = y0.reshape(len(y0),1)
        x = data[:,5:]
        cur_all_input = np.concatenate([x,T],axis=1)
        train_indices,val_indices,test_indices = split_train_val_test(x,0.8,0.05,0.15)
        
        all_yf = len(y1)*[0]
        for i in range(len(y1)):
            if T[i]>0:
                all_yf[i] = y1[i]
            else:
                all_yf[i] = y0[i]

        all_yf = np.array(all_yf)

        all_ycf = len(y0)*[0]
        for i in range(len(y0)):
            if T[i]<1:
                all_ycf[i] = y1[i]
            else:
                all_ycf[i] = y0[i]
        all_ycf = np.array(all_ycf)
        all_yf = (all_yf)
        # norm y
        mean_yf_train = np.mean(all_yf[train_indices])
        std_yf_train = np.std(all_yf[train_indices])
        all_yf[train_indices] = (all_yf[train_indices]-mean_yf_train)/(std_yf_train )
        #print(all_yf)

        mean_yf_val = np.mean(all_yf[val_indices])
        std_yf_val = np.std(all_yf[val_indices])
        all_yf[val_indices] = (all_yf[val_indices]-mean_yf_val)/(std_yf_val)
        #print(all_yf)

        mean_yf_test = np.mean(all_yf[test_indices])
        std_yf_test = np.std(all_yf[test_indices])
        all_yf[test_indices] = (all_yf[test_indices]-mean_yf_test)/(std_yf_test)
        #print(all_yf)


        #print(all_ycf)
        mean_ycf_train = np.mean(all_ycf[train_indices])
        std_ycf_train = np.std(all_ycf[train_indices])
        all_ycf[train_indices] = (all_ycf[train_indices]-mean_ycf_train)/(std_ycf_train )
        #print(all_ycf)

        mean_ycf_val = np.mean(all_ycf[val_indices])
        std_ycf_val = np.std(all_ycf[val_indices])
        all_ycf[val_indices] = (all_ycf[val_indices]-mean_ycf_val)/(std_ycf_val)
        #print(all_ycf)

        mean_ycf_test = np.mean(all_ycf[test_indices])
        std_ycf_test = np.std(all_ycf[test_indices])
        all_ycf[test_indices] = (all_ycf[test_indices]-mean_ycf_test)/(std_ycf_test)
        #print(all_ycf)
        
        cur_ite_true = all_yf - all_ycf
        #print(cur_ite_true[:10])
        cur_ite_true[T<1] = -cur_ite_true[T<1]
        #print(cur_ite_true[:10])


       
        yf= all_yf
        true_ite = cur_ite_true
        val_true_ite = true_ite[val_indices]
        #print("val true ite",val_true_ite)
        val_true_ate = np.mean(true_ite[val_indices])
        test_true_ite = true_ite[test_indices]
        #print("test ite",val_true_ite)
        test_true_ate = np.mean(true_ite[test_indices])

        cur_model = train(Model_name,tf.cast(cur_all_input,tf.float32),tf.cast(yf,tf.float32),train_indices,val_indices,config,config["iterations"] ,config["flag_early_stop"],activation = activation)
        cur_save_model_name = "model"
        cur_save_path = './save_Models/Model_AMZ_neg'+ str(Model_name)[8:-2]   + "_" "repeat_" + str(cur_i)
        os.makedirs(cur_save_path,exist_ok=True)
        save_mymodel(cur_save_path,cur_save_model_name,cur_model)

        cur_val_results = []
        pehe,err_ate = evaluation.evaluate_ate_pehe(cur_model,cur_all_input,val_indices,val_true_ite,val_true_ate)
        cur_val_results.append(pehe)
        cur_val_results.append(err_ate)

        cur_val_results_name = './results/val_results_'+ data_name + str(Model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_val_results,cur_val_results_name)

        cur_test_results = []
        pehe,err_ate = evaluation.evaluate_ate_pehe(cur_model,cur_all_input,test_indices,test_true_ite,test_true_ate)
        cur_test_results.append(pehe)
        cur_test_results.append(err_ate)

        cur_test_results_name = './results/test_results_'+ data_name + str(Model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_test_results,cur_test_results_name)

  


def save_results(save_result,save_name):
	np.save(save_name,save_result) 
	print("saved all results ")



def config_pare(iterations,lr_rate,lr_weigh_decay,flag_early_stop,cross_alpha,use_batch,
    rep_alpha,reg_lambda,flag_norm_gnn,flag_norm_rep,out_dropout,GNN_dropout,rep_dropout,inp_dropout,GL_dropout,
    GL_hidden_shape,head_num_gl,rep_hidden_layer,rep_hidden_shape,GNN_hidden_layer,GNN_hidden_shape,
    head_num_att,out_T_layer,out_C_layer,out_hidden_shape,activation
    ):
    all_configs = []
    cur_activation = activation
    
    config = {}
    config["iterations"] = iterations
    config["lr_rate"] = lr_rate
    config["lr_weigh_decay"] = lr_weigh_decay
    config["flag_early_stop"] = flag_early_stop
    config['cross_alpha'] = cross_alpha
    config['rep_alpha'] = rep_alpha
    config['reg_lambda'] = reg_lambda
    config['flag_norm_gnn'] = flag_norm_gnn
    config['flag_norm_rep'] = flag_norm_rep
    config['out_dropout'] = out_dropout
    config['GNN_dropout'] = GNN_dropout
    config['rep_dropout'] = rep_dropout
    config['inp_dropout'] = inp_dropout
    config['GL_dropout'] = GL_dropout
    config['use_batch'] = use_batch
    config['GL_hidden_shape'] = GL_hidden_shape
    config['head_num_gl'] = head_num_gl
    config['rep_hidden_layer'] = rep_hidden_layer
    config['rep_hidden_shape'] = rep_hidden_shape
    config['GNN_hidden_layer'] = GNN_hidden_layer
    config['GNN_hidden_shape'] = GNN_hidden_shape
    config['head_num_att'] = head_num_att
    config['out_T_layer'] = out_T_layer
    config['out_C_layer'] = out_C_layer
    config['out_hidden_shape'] = out_hidden_shape



    return config,cur_activation

    










