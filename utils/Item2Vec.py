# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
np.random.seed(2019)
random.seed(2019)

class jdata_process():
    def __init__(self):
        self.jdata_action = pd.read_csv("../data/jdata_action.csv")  # 用户行为数据
        self.jdata_product = pd.read_csv("../data/jdata_product.csv")  # 商品数据
        self.jdata_shop = pd.read_csv("../data/jdata_shop.csv")  # 商店数据
        self.jdata_shop = self.jdata_shop.rename(columns={'cate': 'main_cate'})
        self.jdata_user = pd.read_csv("../data/jdata_user.csv")  # 用户数据
        self.jdata_comment = pd.read_csv("../data/jdata_comment.csv") #用户评论数据
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_product, on="sku_id", how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_shop, on='shop_id', how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_user, on='user_id', how='inner')
        self.jdata_action['action_date'] = self.jdata_action.action_time.map(lambda x: str(x)[0:10])

    def get_samples(self, dat):
        sample = self.jdata_action[self.jdata_action.action_date >= dat]
        return sample


def I2v(data,f,L):
    """
    Item2Vec
    :param f: 与user 做交叉的特征名称
    :param flag:
    :param L: embedding后向量维度
    :return:
    """
    print("I2v:", f)
    sentence = []
    dic = {}
    for item in data[['user_id', f]].values:
        try:
            dic[item[0]].append(str(item[1]))
        except:
            dic[item[0]] = [str(item[1])]
    for key in dic:
        sentence.append(dic[key])
    print(len(sentence))
    print('training...')
    random.shuffle(sentence)
    model = Word2Vec(sentence, size=L, window=10, min_count=1, workers=10, iter=10)
    # 对embedding的item进行去重
    print('outputing...')
    values = set(data[f].values)
    w2v = []
    for v in values:
        a = [v]
        a.extend(model[str(v)])
        w2v.append(a)
    # 创建dataframe和embedding向量的columns命名
    out_df = pd.DataFrame(w2v)
    names = [f]
    for i in range(L):
        names.append(names[0] + '_embedding_' + str(L) + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle("../temp/" + f  + "_I2v_" + str(L) + ".pkl")

if __name__=="__main__":

    jdata_process_class = jdata_process()
    # sample collecting
    data = jdata_process_class.get_samples("2018-03-15")
    print(data.head())
    # Item2vec
    for item in ['sku_id','vender_id','brand','shop_id','cate']:
        I2v(data, item, 64)

