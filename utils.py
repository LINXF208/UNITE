import math
import os

import numpy as np 
import tensorflow as tf
import pandas as pd
from tensorflow import keras

import UNITE
import evaluation


def comp_hsic(X, t, s_x=1, s_y=1):
    """ Compute the HSIC. 
    Args:
        X (tf.Tensor): Representation matrix.
        t (tf.Tensor): Treatment assignment vector (binary: 0 or 1).
        p (float): Probability of treatment.

    Returns:
        tf.Tensor: computed HSIC.
    """
    K = GaussianKernelMatrix(X, s_x)
    L = GaussianKernelMatrix(t, s_y)
    m = X.shape[0]

    H = tf.raw_ops.MatrixDiag(diagonal=tf.ones(shape=[m, ])) - 1 / m
    LH = tf.matmul(L,H)
    HLH = tf.matmul(H,LH)
    KHLH = tf.matmul(K,HLH)
    HSIC = tf.linalg.trace(KHLH)/((m-1)**2)

    return HSIC


def GaussianKernelMatrix(x, sigma=1):
    """ Computes the Gaussian Kernel Matrix"""
    pairwise_distances = pdist2sq(x, x) # Computes the squared Euclidean distance
    return tf.exp(-pairwise_distances / 2 * sigma)


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = (C + tf.transpose(ny)) + nx

    return D    


def divide_t_c(concated_data, input_t):
    """Divide units into Treated and Control groups.

    Args:
        concated_data (tf.Tensor): The dataset containing all units.
        input_t (tf.Tensor): Binary tensor (0: Control, 1: Treated).

    Returns:
        tuple: (group_T, group_C, i0, i1)
            - group_T (tf.Tensor): Treated group data.
            - group_C (tf.Tensor): Control group data.
            - i0 (tf.Tensor): Indices of the Control group.
            - i1 (tf.Tensor): Indices of the Treated group.
    """
    i0 = tf.cast((tf.where(input_t < 1)[:, 0]), tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:, 0]), tf.int32)

    group_T = tf.gather(concated_data, i1)
    group_C = tf.gather(concated_data, i0)

    return group_T, group_C, i0, i1


def split_train_val_test(data, train_ratio, val_ratio, test_ratio,seed=42):
    """
    Split data indices into training, validation, and test sets.

    Args:
        data (array-like): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))

    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)

    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]
    
    return train_indices, val_indices, test_indices


def train(
        model_name, 
        input_data,y, 
        train_idx, 
        val_idx, 
        config, 
        max_iterations, 
        flag_early_stop=False, 
        activation=tf.nn.relu
    ):
    """
    Train process.

    Args:
        model_name (tf.keras.Model): The model class to be trained.
        input_data (np.array): Input data of all units.
        train_idx, val_idx (list): Indices for training, validation set, respectively
        config (dict): Model hyperparameters.
        max_iterations (int): Maximum number of training iterations.
        flag_early_stop (bool): Whether to enable early stopping.
        activation (function): Activation function used in the model.

    Returns:
        tf.keras.Model: The trained model.
    """
    cur_all_features = input_data[:, :-1]
    cur_model = model_name(config, activation=activation)

    loss_list_val = []
    losslist = []
    count = 0
    sum_loss = 0
    sum_val_loss = 0
    
    for i in range(max_iterations):
        print("iter", i)

        loss = cur_model.val_y(input_data, tf.cast(y, tf.float32), train_idx)
        total_loss = cur_model.network_learn(tf.cast(input_data, tf.float32), tf.cast(y, tf.float32), train_idx)
        val_loss = cur_model.val_y(tf.cast(input_data, tf.float32), tf.cast(y, tf.float32), val_idx)

        sum_loss += loss
        sum_val_loss += val_loss

        if (i+1) % 20 == 0:
            if len(loss_list_val) > 0 and sum_val_loss/20 >= loss_list_val[-1]:
                count += 1
            else:
                count = 0

            if flag_early_stop:
                if i > 400 and count >= 1:
                    break

            losslist.append(sum_loss / 20)
            loss_list_val.append(sum_val_loss / 20)

            sum_loss = 0
            sum_val_loss = 0

    return cur_model

 
def save_my_model(save_path, save_name, need_save_model):
    """
    Save the model weights to a specified path.

    Args:
        save_path (str): Directory to save the model weights.
        save_name (str): Filename for the saved weights.
        need_save_model (tf.keras.Model): Model instance to be saved.

    Returns:
        None
    """
    path = save_path + '/' + save_name

    need_save_model.save_weights(path)
    print("Already saved the model's weights in file" + path)


def load_my_model(load_path, load_name, need_load_model, config, activation):
    """
    Load a saved model from a specified path.

    Args:
        load_path (str): Directory where the model is saved.
        load_name (str): Filename of the saved model.
        need_load_model (tf.keras.Model): Model class to instantiate.
        config (dict): Model configuration parameters.
        activation (tf activation function): Activation function.

    Returns:
        tf.keras.Model: Loaded model instance.
    """
    model = need_load_model(config, activation)

    path = load_path + '/' + load_name

    model.load_weights(path)
    print("Model successfully loaded.")

    return model


def implement_UNITE(config, data_name, model_name, activation, cur_i):
    """ 
    Train, evaluate, and save model with results.

    Args:
        config (dict): Hyperparameters.
        data_name (str): Dataset name.
        model_name (class): Model class.
        activation (function): Activation function.
        cur_i (int): (cur_i)-th repeated run. 
    """
    if data_name == 'AMZ_neg':
        load_datas = pd.read_csv("./data/AmazonItmFeatures_neg.csv",header=None, prefix="col")
        data = load_datas.to_numpy()

        x = data[:, 5:]
        t = data[:, 0]
        y1 = data[:, 1]
        y0 = data[:, 2]
        t = t.reshape(len(t), 1)
        y1 = y1.reshape(len(y1), 1)
        y0 = y0.reshape(len(y0), 1)

        cur_all_input = np.concatenate([x, t], axis=1)

        train_indices, val_indices, test_indices = split_train_val_test(x, 0.8, 0.05, 0.15)
        
        all_yf = len(y1) * [0]
        for i in range(len(y1)):
            if t[i] > 0:
                all_yf[i] = y1[i]
            else:
                all_yf[i] = y0[i]
        all_yf = np.array(all_yf)

        all_ycf = len(y0) * [0]
        for i in range(len(y0)):
            if t[i] < 1:
                all_ycf[i] = y1[i]
            else:
                all_ycf[i] = y0[i]
        all_ycf = np.array(all_ycf)

        mean_yf_train = np.mean(all_yf[train_indices])
        std_yf_train = np.std(all_yf[train_indices])
        all_yf[train_indices] = (all_yf[train_indices] - mean_yf_train) / std_yf_train

        mean_yf_val = np.mean(all_yf[val_indices])
        std_yf_val = np.std(all_yf[val_indices])
        all_yf[val_indices] = (all_yf[val_indices] - mean_yf_val) / std_yf_val

        mean_yf_test = np.mean(all_yf[test_indices])
        std_yf_test = np.std(all_yf[test_indices])
        all_yf[test_indices] = (all_yf[test_indices] - mean_yf_test) / std_yf_test

        mean_ycf_train = np.mean(all_ycf[train_indices])
        std_ycf_train = np.std(all_ycf[train_indices])
        all_ycf[train_indices] = (all_ycf[train_indices] - mean_ycf_train) / std_ycf_train

        mean_ycf_val = np.mean(all_ycf[val_indices])
        std_ycf_val = np.std(all_ycf[val_indices])
        all_ycf[val_indices] = (all_ycf[val_indices] - mean_ycf_val) / std_ycf_val

        mean_ycf_test = np.mean(all_ycf[test_indices])
        std_ycf_test = np.std(all_ycf[test_indices])
        all_ycf[test_indices] = (all_ycf[test_indices] - mean_ycf_test) / std_ycf_test

        true_ite = all_yf - all_ycf
        true_ite[t < 1] = -true_ite[t < 1]

        all_input = tf.cast(all_input, tf.float32)
        yf = tf.cast(all_yf, tf.float32)

        val_true_ite = true_ite[val_indices]
        val_true_ate = np.mean(true_ite[val_indices])
        test_true_ite = true_ite[test_indices]
        test_true_ate = np.mean(true_ite[test_indices])

        model = train(
            model_name, 
            all_input, 
            yf, 
            train_indices, 
            val_indices, 
            config, 
            config["iterations"], 
            config["flag_early_stop"], 
            activation=activation
        )

        os.makedirs(cur_save_path, exist_ok=True)

        val_results = []
        test_results = []

        cur_save_model_name = "model"
        cur_save_path = './save_Models/Model_AMZ_neg_' + str(model_name)[8:-2] + "_" + "repeat_" + str(cur_i)
        save_my_model(cur_save_path, cur_save_model_name, model)

        val_pehe, val_err_ate = evaluation.evaluate_ate_pehe(model, all_input, val_indices, val_true_ite, val_true_ate)
        val_results = [val_pehe, val_err_ate]
        val_results_name = './results/val_results_' + data_name + '_' + str(model_name)[8:-2] + '_' + "reapted_" + str(cur_i)
        save_results(val_results, val_results_name)

        test_pehe, test_err_ate = evaluation.evaluate_ate_pehe(model, all_input, test_indices, test_true_ite, test_true_ate)
        test_results.append(test_pehe, test_err_ate)
        test_results_name = './results/test_results_' + data_name + '_' + str(model_name)[8:-2] + '_' + "reapted_" + str(cur_i)
        save_results(test_results, test_results_name)


def save_results(save_result, save_name):
    """
    Save results as a .npy file in the specified directory.

    Args:
        save_result (numpy array): Data to save.
        save_path (str): Directory where the results should be saved.
        save_name (str): Filename for saving the results.

    Returns:
        None
    """
    np.save(save_name, save_result) 
    print("saved all results ")


def config_pare(
        iterations,
        lr_rate,
        lr_weigh_decay,
        flag_early_stop,
        cross_alpha,
        use_batch,
        rep_alpha,
        reg_lambda,
        flag_norm_gnn,
        flag_norm_rep,
        out_dropout,
        GNN_dropout,
        rep_dropout,
        inp_dropout,
        GL_dropout,
        GL_hidden_shape,
        head_num_gl,
        rep_hidden_layer,
        rep_hidden_shape,
        GNN_hidden_layer,
        GNN_hidden_shape,
        head_num_att,
        out_T_layer,
        out_C_layer,
        out_hidden_shape,
    ):
    """Generate a configuration dictionary for model parameters."""

    config = {
        "iterations": iterations,
        "lr_rate": lr_rate,
        "lr_weigh_decay": lr_weigh_decay,
        "flag_early_stop": flag_early_stop,
        "cross_alpha": cross_alpha,
        "rep_alpha": rep_alpha,
        "reg_lambda": reg_lambda,
        "flag_norm_gnn": flag_norm_gnn,
        "flag_norm_rep": flag_norm_rep,
        "out_dropout": out_dropout,
        "GNN_dropout": GNN_dropout,
        "rep_dropout": rep_dropout,
        "inp_dropout": inp_dropout,
        "GL_dropout": GL_dropout,
        "use_batch": use_batch,
        "GL_hidden_shape": GL_hidden_shape,
        "head_num_gl": head_num_gl,
        "rep_hidden_layer": rep_hidden_layer,
        "rep_hidden_shape": rep_hidden_shape,
        "GNN_hidden_layer": GNN_hidden_layer,
        "GNN_hidden_shape": GNN_hidden_shape,
        "head_num_att": head_num_att,
        "out_T_layer": out_T_layer,
        "out_C_layer": out_C_layer,
        "out_hidden_shape": out_hidden_shape,
    }

    return config

    










