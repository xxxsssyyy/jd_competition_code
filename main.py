# -*- coding: utf-8 -*-
from utils.ReadData import *
from utils.DeepFM import DeepFM
import tensorflow as tf
import numpy as np
import pandas as pd
import gc


if __name__ == '__main__':
    # 输入模型中的连续特征
    numeric_cols = ['sku_good_comment_rate','sku_bad_comment_rate']
    # 输入模型中的离散特征
    cate_cols = ['user_id','sku_id','module_id','type','brand','shop_id','cate',
                 'vender_id','city_level','province','city','county']
    fd = FeatureDictionary(numeric_cols,cate_cols)
    data_parser = DataParser(feat_dict=fd)

    Xi_train, Xv_train, Xo_train, y_train = data_parser.parse(df=fd.dfTrain, has_label=True)
    Xi_test, Xv_test, Xo_test, ids_test = data_parser.parse(df=fd.dfTest, has_label=False)

    # 最底层field取值
    field_lens = [fd.df[col].unique() for col in [numeric_cols+cate_cols]]
    # params
    dfm_params = {
        "vec_dim": 16,
        "field_lens": field_lens,
        "lr": 0.001,
        "dropout_rate": 0.05,
        "i2v_num": 320,
        'batch_size':64,
        'eqoch':5,
        "verbose": True
    }
    dfm = DeepFM(**dfm_params)
    dfm.fit(Xi_train, Xv_train, Xo_train, y_train)

    y_test_meta = np.zeros((fd.dfTest.shape[0], 1), dtype=float)
    y_test_meta[:, 0] = dfm.predict(Xi_test, Xv_test, Xo_test)

    fd.dfTest['predict'] = y_test_meta
    # 结果去重
    result = fd.dfTest[['user_id', 'cate', 'shop_id']].drop_duplicates()
    #result.to_csv("./temp/result.csv", index=None)