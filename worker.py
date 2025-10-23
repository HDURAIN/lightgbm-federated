import torch.nn.functional as F
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None
import torch
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.utils.data
import os
import joblib
from lightgbm import log_evaluation, early_stopping

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=5)]
train_datasets_dir = './data'
test_dataset_dir = './data/test.csv'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


def get_user_data(user_idx):
    # get user's data
    fpath = ""
    for root, dirs, fnames in os.walk(train_datasets_dir):
        fname = fnames[user_idx]
        print(fname)
        fpath = os.path.join(root, fname)
        break

    if not fpath.endswith('csv'):
        return
    print('Load User {} Data: '.format(user_idx), os.path.basename(fpath))
    data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
    # print('he')
    # print(data)
    return data


class Worker(object):
    def __init__(self,user_idx):
        self.user_idx = user_idx
        self.data = get_user_data(self.user_idx)
        self.params = {
                'task': 'train',
                'boosting_type': 'gbdt',    # 设置提升类型
                'objective': 'multiclass',  # 目标函数
                'num_leaves': 40,           # 叶子节点数
                'learning_rate': 0.05,      # 学习速率
                'feature_fraction': 0.9,    # 建树的特征选择比例
                'bagging_fraction': 0.8,    # 建树的样本采样比例
                'bagging_freq': 5,          # k 意味着每 k 次迭代执行bagging
                'verbose': 1,               # =0 显示错误 (警告), >0 显示信息
                'num_class': 14,            # 类别
        }
        # print(type(self.data))
        # print(self.data)
        self.lgb_train, self.lgb_eval = self.preprocess_data()

    def preprocess_data(self):
        '''
        Args:
            data——DataFrame
        Return:
            lgb_Dataset
        '''
        y = np.array([
            ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
            for t in self.data.iloc[:, -1]
        ])
        y = y.ravel()
        # print(y.shape)
        x = self.data.drop(['Label'], axis=1).select_dtypes(exclude=['object'])
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x.values, y, test_size=0.2)
        lgb_train = lgb.Dataset(self.train_x, self.train_y)
        lgb_eval = lgb.Dataset(self.test_x, self.test_y, reference=lgb_train)
        return lgb_train, lgb_eval

    def user_round_train_eval(self):
        # train lightgbm
        self.model = lgb.train(self.params, self.lgb_train, num_boost_round=60, valid_sets=self.lgb_eval,callbacks=callbacks)

        #self.model = lgb.cv(self.params, self.lgb_train, num_boost_round=60, nfold=5, callbacks=callbacks)
        predictions = self.model.predict(self.test_x, num_iteration=self.model.best_iteration)
        # self.model.save_model('model' + str(self.user_idx) + '.txt')
        y_pred = [list(x).index(max(x)) for x in predictions]
        print('The accuracy of prediction in eval dataset is:' + str(accuracy_score(self.test_y, y_pred)))

    def save_model(self):
        joblib.dump(self.model, 'loan_model' + str(self.user_idx) + '.pkl')

    def user_round_prediction(self, dataset):
        # 这里的dataset是指x，且x为lgb.Dataset
        predictions = self.model.predict(dataset, num_iteration=self.model.best_iteration)
        y_pred = [list(x).index(max(x)) for x in predictions]
        return y_pred

