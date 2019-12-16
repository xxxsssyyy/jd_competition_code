# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

class DeepFM(object):
    def __init__(self, vec_dim=None, field_lens=None, dnn_layers=[32, 32],
                 lr=0.001, dropout_rate=None, i2v_num = None,optimizer_type="adam",
                 eval_metric=f1_score, batch_size=128,epoch = 5,verbose=False):

        self.vec_dim = vec_dim # field 的embedding vector维度
        self.field_lens = field_lens
        # list结构，其中每个元素代表对应Field有多少取值。例如gender有两个取值，那么其元素为2,不用考虑顺序
        self.field_num = len(field_lens)
        self.feat_num = np.sum(field_lens)
        # dnn_layers：list结构，其中每个元素对应DNN部分节点数目
        self.dnn_layers = dnn_layers
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.optimizer_type = optimizer_type
        self.eval_metric = eval_metric
        self.batch_size=batch_size
        self.epoch = epoch
        self.verbose = verbose
        self.train_result, self.valid_result = [], []

        self.I2v_num = i2v_num # 64 * 5 =320

        self._build_graph()

    def _build_graph(self):
        self.add_input()
        self.inference()

    def add_input(self):
        self.index = tf.placeholder(tf.int32, shape=[None, self.field_num], name='feat_index')  # (batch, F)
        self.x = tf.placeholder(tf.float32, shape=[None, self.field_num], name='feat_value')  # (batch, F)
        # Item2Vec Emb
        self.I2v_Emb = tf.placeholder(tf.float32, shape=[None, self.I2v_num], name='I2v_Emb')  # (batch, I)
        self.y = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.is_train = tf.placeholder(tf.bool)

    def inference(self):
        with tf.variable_scope('first_order_part'):
            first_ord_w = tf.get_variable(name='first_ord_w', shape=[self.feat_num, 1], dtype=tf.float32)
            first_order = tf.nn.embedding_lookup(first_ord_w, self.index)  # (batch, F, 1)
            first_order = tf.reduce_sum(tf.multiply(first_order, tf.expand_dims(self.x, axis=2)), axis=2)  # (batch, F)
        with tf.variable_scope('emb_part'):
            embed_matrix = tf.get_variable(name='second_ord_v', shape=[self.feat_num, self.vec_dim], dtype=tf.float32)
            embed_v = tf.nn.embedding_lookup(embed_matrix, self.index)  # (batch, F, K)
            embed_x = tf.multiply(tf.expand_dims(self.x, axis=2), embed_v)  # (batch, F, K)
        with tf.variable_scope('second_order_part'):
            sum_emb_square = tf.square(tf.reduce_sum(embed_x, axis=1))  # (batch, K)
            square_emb_sum = tf.reduce_sum(tf.square(embed_x), axis=1)  # (batch, K)
            second_order = 0.5 * (sum_emb_square - square_emb_sum)
            fm = tf.concat([first_order, second_order], axis=1)  # (batch, F+K)
        with tf.variable_scope('dnn_part'):
            embed_x = tf.layers.dropout(embed_x, rate=self.dropout_rate, training=self.is_train)  # (batch, F, K)
            in_num_1 = self.field_num * self.vec_dim
            dnn_1 = tf.reshape(embed_x, shape=(-1, in_num_1))  # (batch, in_num1)
            dnn  = tf.concat([dnn_1,self.I2v_Emb],axis=1) # (batch, in_num) = (batch,F*K+I)
            in_num = in_num_1 + self.I2v_num
            for i in range(len(self.dnn_layers)):
                out_num = self.dnn_layers[i]
                w = tf.get_variable(name='w_%d' % i, shape=[in_num, out_num], dtype=tf.float32)
                b = tf.get_variable(name='b_%d' % i, shape=[out_num], dtype=tf.float32)
                dnn = tf.matmul(dnn, w) + b
                dnn = tf.layers.dropout(tf.nn.relu(dnn), rate=self.dropout_rate, training=self.is_train)
                in_num = out_num
        with tf.variable_scope('output_part'):
            in_num += self.field_num + self.vec_dim # in_num = F+K+self.dnn_layers[-1]
            output = tf.concat([fm, dnn], axis=1)
            proj_w = tf.get_variable(name='proj_w', shape=[in_num, 1], dtype=tf.float32)
            proj_b = tf.get_variable(name='proj_b', shape=[1], dtype=tf.float32)
            self.y_logits = tf.matmul(output, proj_w) + proj_b

        self.y_hat = tf.nn.sigmoid(self.y_logits)
        self.pred_label = tf.cast(self.y_hat > 0.5, tf.int32)
        # loss
        self.loss = -tf.reduce_mean(self.y * tf.log(self.y_hat + 1e-8) + (1 - self.y) * tf.log(1 - self.y_hat + 1e-8))

        #self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        # optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        elif self.optimizer_type == "momentum":
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.95).minimize(
                self.loss)

        # init
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = self._init_session()
        self.sess.run(init)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self, Xi, Xv, Xo,y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end],Xo[start:end], [[y_] for y_ in y[start:end]]

    def shuffle_in_unison_scary(self, a, b, c, d):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)

    def fit_on_batch(self, Xi, Xv, Xo, y):
        feed_dict = {self.index: Xi,
                     self.x: Xv,
                     self.I2v_Emb: Xo,
                     self.y: y,
                     self.is_train: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, Xi_train, Xv_train,Xo_train, y_train,
            Xi_valid=None, Xv_valid=None, Xo_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float
        :param Xo_train: [[],[],[],[]...[]]
        :param y_train: label of each sample in the training set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train ,Xo_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, Xo_batch, y_batch = self.get_batch(Xi_train, Xv_train, Xo_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, Xo_train, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(Xi_train, Xv_train,Xo_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(Xi_valid, Xv_valid,Xo_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping:
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:

            best_valid_score = max(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            # 全量数据再训练
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            Xo_train = Xo_train + Xo_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, Xo_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, Xo_train, y_batch = self.get_batch(Xi_train, Xv_train, Xo_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, Xo_train, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, Xo_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        ( train_result > best_train_score) :
                    break

    def predict(self, Xi, Xv ,Xo):
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch,Xo_batch, y_batch = self.get_batch(Xi, Xv, Xo, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.index: Xi_batch,
                         self.x: Xv_batch,
                         self.I2v_Emb : Xo_batch,
                         self.y: y_batch,
                         self.is_train: False}
            batch_out = self.sess.run(self.pred_label, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch,Xo_batch, y_batch = self.get_batch(Xi, Xv,Xo, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, Xo, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(Xi, Xv, Xo)
        return self.eval_metric(y, y_pred)

