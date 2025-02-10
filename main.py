import math

import numpy as np 
import tensorflow as tf
from tensorflow import keras

import UNITE
import utils
import ourlayers
import evaluation


def main(cur):
    configs = utils.config_pare(
        iterations=2000, 
        lr_rate=0.001, 
        lr_weigh_decay=0.001, 
        flag_early_stop=True, 
        use_batch=256,
        rep_alpha=2.0,
        reg_lambda=2e-2,
        flag_norm_gnn=False,
        flag_norm_rep=False,
        out_dropout=0.5,
        GNN_dropout=0.1,
        rep_dropout=0.1,
        inp_dropout=0.0,
        GL_dropout=0.1,
        GL_hidden_shape=[64],
        head_num_gl=1,
        rep_hidden_layer=3,
        rep_hidden_shape=[128, 64, 64],
        GNN_hidden_layer=3,
        GNN_hidden_shape=[64, 64, 32],
        head_num_att=8,
        out_T_layer=3,
        out_C_layer=3,
        out_hidden_shape=[128, 64, 32],
        cross_alpha=1.5
    )

    utils.implement_UNITE(
        config=configs,
        data_name='AMZ_neg',
        Model_name=UNITE.UNITE,
        activation=tf.nn.relu,
        cur_i=cur
    )


if __name__ == '__main__':
    for rep_num in range(0, 3):
        main(rep_num)

