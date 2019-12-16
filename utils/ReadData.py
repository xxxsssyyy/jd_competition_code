# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import datetime
import gc
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

class jdata_process():
    def __init__(self):
        self.jdata_action = pd.read_csv("./data/jdata_action.csv")  # 用户行为数据
        self.jdata_product = pd.read_csv("./data/jdata_product.csv")  # 商品数据
        self.jdata_shop = pd.read_csv("./data/jdata_shop.csv")  # 商店数据
        self.jdata_shop = self.jdata_shop.rename(columns={'cate': 'main_cate'})
        self.jdata_user = pd.read_csv("./data/jdata_user.csv")  # 用户数据
        self.jdata_comment = pd.read_csv("./data/jdata_comment.csv") #用户评论数据
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_product, on="sku_id", how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_shop, on='shop_id', how='inner')
        self.jdata_action = pd.merge(self.jdata_action, self.jdata_user, on='user_id', how='inner')
        self.jdata_action['action_date'] = self.jdata_action.action_time.map(lambda x: str(x)[0:10])

        # 同一商品总评论数、总好评数、总坏评数
        sku_comment_data = self.jdata_comment.groupby('sku_id').sum().reset_index().rename(columns={
            'comments': 'sku_comment_nums',
            'good_comments': 'sku_good_comment_nums',
            'bad_comments': 'sku_bad_comment_nums'
        })
        # 商品总好评率、坏评率
        mean_good_comment_rate = sum(sku_comment_data.sku_good_comment_nums) * 1.0 / sum(
            sku_comment_data.sku_comment_nums)
        mean_bad_comment_rate = sum(sku_comment_data.sku_bad_comment_nums) * 1.0 / sum(
            sku_comment_data.sku_comment_nums)
        # 同一商品平滑的好评率、坏评率
        sku_comment_data['sku_good_comment_rate'] = (sku_comment_data[
                                                         'sku_good_comment_nums'] + 50 * mean_good_comment_rate) / (
                                                            sku_comment_data['sku_comment_nums'] + 50)
        sku_comment_data['sku_bad_comment_rate'] = (sku_comment_data[
                                                        'sku_bad_comment_nums'] + 50 * mean_bad_comment_rate) / (
                                                           sku_comment_data['sku_comment_nums'] + 50)
        self.jdata_action = pd.merge(self.jdata_action, sku_comment_data, on='sku_id', how='left')


    # 获得样本 三天内有购买
    def get_samples(self, samples, dat, dat2):
        sample = samples[samples.action_date == dat]
        target = self.jdata_action[(self.jdata_action.type == 2) & (self.jdata_action.action_date > dat) & (
                    self.jdata_action.action_date <= dat2)].groupby(
            ['user_id', 'cate', 'shop_id']).size().reset_index().rename(columns={0: 'flag'})
        sample = pd.merge(sample, target, on=['user_id', 'cate', 'shop_id'], how='left')
        sample = sample.fillna(0)
        # 给有购买的正样本标签为1，无购买样本标签为0
        sample['flag'] = sample.flag.map(lambda x: 1 if x >= 1 else 0)
        return sample

    def gen_samples(self):
        # 去重
        samples = self.jdata_action.groupby(['user_id', 'cate', 'shop_id', 'action_date']).size().reset_index()[
            ['user_id', 'cate', 'shop_id', 'action_date']]
        # 设置三天内购买为正样本
        samples0408 = self.get_samples(samples, "2018-04-08", "2018-04-11")
        samples0409 = self.get_samples(samples, "2018-04-09", "2018-04-12")
        samples0410 = self.get_samples(samples, "2018-04-10", "2018-04-13")
        samples0411 = self.get_samples(samples, "2018-04-11", "2018-04-14")
        samples0412 = self.get_samples(samples, "2018-04-12", "2018-04-15")
        submit_sample1 = self.get_samples(samples, "2018-04-15", "2018-04-18")
        submit_sample2 = self.get_samples(samples, "2018-04-14", "2018-04-17")
        submit_sample3 = self.get_samples(samples, "2018-04-13", "2018-04-16")
        return pd.concat(
            [samples0408, samples0409, samples0410, samples0411, samples0412, submit_sample1, submit_sample2,
             submit_sample3])

    def add_static_fea(self,sample):
        # 加入静态属性特征
        sample = pd.merge(sample, self.jdata_shop, on='shop_id', how='left')
        sample = pd.merge(sample, self.jdata_user, on='user_id', how='left')
        sample = sample[~sample.shop_reg_tm.isnull()]
        sample['shop_reg_year'] = sample.shop_reg_tm.map(lambda x: int(str(x)[0:4]))
        sample['shop_reg_days'] = sample.shop_reg_tm.map(lambda x: np.NAN if str(x) == 'nan' else (datetime.date(2019, 1, 1) - datetime.date(int(str(x)[0:4]), int(str(x)[5:7]),int(str(x)[8:10]))).days)
        sample['same_with_main_cate'] = (sample.main_cate == sample.cate).map(lambda x: 1 if x else 0)
        sample['user_reg_year'] = sample.user_reg_tm.map(lambda x: int(str(x)[0:4]))
        sample['user_reg_days'] = sample.user_reg_tm.map(lambda x: np.NAN if str(x) == 'nan' else (datetime.date(2019, 1, 1) - datetime.date(int(str(x)[0:4]), int(str(x)[5:7]),int(str(x)[8:10]))).days)
        return sample

    def add_comment_fea(self,sample):
        # shop static feature
        # 刻画该商店下商品数量、品牌名数量、品种数量
        shop_own_features = self.jdata_product.groupby('shop_id')['sku_id', 'brand', 'cate'].nunique().reset_index().rename(
            columns={
                'sku_id': 'shop_own_skus',
                'brand': 'shop_own_brands',
                'cate': 'shop_own_cates'
            })
        # 刻画该商店评论总数、好评总数、坏评总数
        shop_comment_fea1 = self.jdata_product.groupby('shop_id')[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'shop_comment_nums',
            'sku_good_comment_nums': 'shop_good_comment_nums',
            'sku_bad_comment_nums': 'shop_bad_comment_nums'
        })
        # 该商店平均好评比例、该商店商品最高、最低好评率
        shop_comment_fea2 = self.jdata_product.groupby('shop_id')['sku_good_comment_rate', 'sku_bad_comment_rate'].agg(
            ['mean', 'max', 'min']).reset_index()
        shop_comment_fea2.columns = shop_comment_fea2.columns.droplevel(0)
        shop_comment_fea2.columns = ['shop_id', 'shop_good_comm_rate_mean', 'shop_good_comm_rate_max',
                                     'shop_good_comm_rate_min', \
                                     'shop_bad_comm_rate_mean', 'shop_bad_comm_rate_max', 'shop_bad_comm_rate_min']

        # cate static feature
        # 该品种下商品数，品牌数，商店数
        cate_own_features = self.jdata_product.groupby('cate')['sku_id', 'brand', 'shop_id'].nunique().reset_index().rename(
            columns={
                'sku_id': 'cate_own_skus',
                'brand': 'cate_own_brands',
                'shop_id': 'cate_own_shops'
            })
        # 该品种下商品评论总数、好评总数、差评总数
        cate_comment_fea1 = self.jdata_product.groupby('cate')[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'cate_comment_nums',
            'sku_good_comment_nums': 'cate_good_comment_nums',
            'sku_bad_comment_nums': 'cate_bad_comment_nums'
        })
        # 该品种下商品平均好评率、最高最低好评率;该品种下商品平均差评率、最高最低差评率
        cate_comment_fea2 = self.jdata_product.groupby('cate')['sku_good_comment_rate', 'sku_bad_comment_rate'].agg(
            ['mean', 'max', 'min']).reset_index()
        cate_comment_fea2.columns = cate_comment_fea2.columns.droplevel(0)
        cate_comment_fea2.columns = ['cate', 'cate_good_comm_rate_mean', 'cate_good_comm_rate_max',
                                     'cate_good_comm_rate_min',
                                     'cate_bad_comm_rate_mean', 'cate_bad_comm_rate_max', 'cate_bad_comm_rate_min']
        # shop and cate  And Sort Features like which is the most popular cate in a shop
        # 该商店品种下用户数和品牌数
        shop_cate_own_features = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_id', 'brand'].nunique().reset_index().rename(columns={
            'sku_id': 'shop_cate_own_skus',
            'brand': 'shop_cate_own_brands'
        })
        # 该商店品种下商品评论总数、好评总数、差评总数
        shop_cate_comment_fea1 = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_comment_nums', 'sku_good_comment_nums', 'sku_bad_comment_nums'].sum().reset_index().rename(columns={
            'sku_comment_nums': 'shop_cate_comment_nums',
            'sku_good_comment_nums': 'shop_cate_good_comment_nums',
            'sku_bad_comment_nums': 'shop_cate_bad_comment_nums'
        })
        # 该商店品种下商品平均好评率、最高最低好评率；该商店品种下商品平均差评率、最高最低差评率
        shop_cate_comment_fea2 = self.jdata_product.groupby(['shop_id', 'cate'])[
            'sku_good_comment_rate', 'sku_bad_comment_rate'].agg(['mean', 'max', 'min']).reset_index()
        shop_cate_comment_fea2.columns = shop_cate_comment_fea2.columns.droplevel(0)
        shop_cate_comment_fea2.columns = ['shop_id', 'cate', 'shop_cate_good_comm_rate_mean',
                                          'shop_cate_good_comm_rate_max', 'shop_cate_good_comm_rate_min',
                                          'shop_cate_bad_comm_rate_mean', 'shop_cate_bad_comm_rate_max',
                                          'shop_cate_bad_comm_rate_min']

        shop_static_fea = pd.merge(shop_own_features, shop_comment_fea1, on='shop_id', how='outer')
        shop_static_fea = pd.merge(shop_static_fea, shop_comment_fea2, on='shop_id', how='outer')
        shop_static_fea = shop_static_fea.fillna(shop_static_fea.mean())
        cate_static_fea = pd.merge(cate_own_features, cate_comment_fea1, on='cate', how='outer')
        cate_static_fea = pd.merge(cate_static_fea, cate_comment_fea2, on='cate', how='outer')
        cate_static_fea = cate_static_fea.fillna(cate_static_fea.mean())
        shop_cate_static_fea = pd.merge(shop_cate_own_features, shop_cate_comment_fea1, on=['shop_id', 'cate'],
                                        how='outer')
        shop_cate_static_fea = pd.merge(shop_cate_static_fea, shop_cate_comment_fea2, on=['shop_id', 'cate'],
                                        how='outer')
        shop_cate_static_fea = shop_cate_static_fea.fillna(shop_cate_static_fea.mean())

        sample = pd.merge(sample, shop_static_fea, on='shop_id')
        sample = pd.merge(sample, cate_static_fea, on='cate')
        sample = pd.merge(sample, shop_cate_static_fea, on=['shop_id', 'cate'])
        return sample

    def join_item2Vec(self):
        # 假如Emb的特征
        sample = self.gen_samples()
        for item in ['vender_id','brand','shop_id','cate']:
            temp = pd.read_pickle("../temp/" + item  + "_I2v_" + "64" + ".pkl")
            sample = pd.merge(sample,temp,on=item,how='left')
        return sample

class FeatureDictionary(object):
    def __init__(self,  numeric_cols=[], cate_cols=[]):
        self.dfTrain = None
        self.dfTest = None
        self.numeric_cols = numeric_cols
        self.cate_cols = cate_cols
        self.data = jdata_process().join_item2Vec() # 在这里将Item2Vec进行join
        self.gen_feat_dict()

    def gen_feat_dict(self):
        self.dfTrain = self.data[self.data.action_date < '2018-04-13']

        self.dfTest = self.data[(self.data.action_date>='2018-04-13')&(self.data.action_date<'2018-04-15')]
        self.df = pd.concat([self.dfTrain, self.dfTest])
        self.feat_dict = {}
        tc = 0
        for col in self.df.columns:
            if col in self.cate_cols:
                us = self.df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                continue
        self.feat_dim = tc

class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, df=None, has_label=False):
        dfi = df.copy()
        dfo = df.copy()
        if has_label:
            y = dfi["flag"].values.tolist()
            dfi.drop(['sku_id','vender_id','brand','shop_id','cate',"flag"], axis=1, inplace=True)
        else:
            ids = dfi['sku_id'].values.tolist()
            dfi.drop(['sku_id','vender_id','brand','shop_id','cate'], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.cate_cols:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()

        # I2v列名
        I2v_cols=[]
        for col in ['vender_id','brand','shop_id','cate']:
            for i in range(64):
                I2v_cols.append(col+'_embedding_64_'+str(i))
        Xo = dfo[I2v_cols].values.tolist()
        if has_label:
            return Xi, Xv, Xo, y
        else:
            return Xi, Xv, Xo, ids
