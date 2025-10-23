import numpy as np
import pandas as pd
import os
import sys
import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from worker import Worker
from pandas.core.frame import DataFrame
pd.set_option('display.max_columns', None)

train_datasets_dir = '/train.csv'        # x个数据集，x个用户
test_dataset_dir = './data/test.csv'
# model_save_path = '/Users/klaus_imac/Desktop/Tree/model'
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
num_user = 1
for root, dirs, files in os.walk(train_datasets_dir):
    for each in files:
        if 'DS' in each:
            continue
        else:
            num_user += 1
#print('The number of users: {}'.format(num_user))

# 计算总耗时s1
s1_start = datetime.datetime.now()

# 设立多个用户，分配dataset
workers = []
# for u in range(1, num_user+1):
# 这里的时间s2应该忽略不计
s2_start = datetime.datetime.now()
for u in range(1, num_user+1):
    workers.append(Worker(user_idx=u))
s2_end = datetime.datetime.now()
s2 = (s2_end - s2_start).seconds
#print('S2 Time cost: ' + str(s2))

# 每个用户自己训练
print('Start training:')
# 这里的时间s3应该是并行的
s3_start = datetime.datetime.now()
for u in range(num_user):
    print('-------------------------------------')
    #print('This is the ' + str(u) + 'th User:')
    workers[u].user_round_train_eval()
    #print('Time cost: ')
s3_end = datetime.datetime.now()
s3 = (s3_end - s3_start).seconds
#print('S3 Time cost: ' + str(s3))

# 将训练好的模型保存到中心服务器上
# for u in range(num_user):
#     workers[u].save_model()

# 测试集预处理
test_data = pd.read_csv(test_dataset_dir)
#test_data =test_data.head(5)#取前n条数据
#print(test_data.head())
#x = test_data.drop(['Label'], axis=1).select_dtypes(exclude=['object'])
y = np.array([
            ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
            for t in test_data.iloc[:, -1]
        ])
y = y.ravel()
#print(y.shape)
x = test_data.iloc[:, :-1].select_dtypes(exclude=['object'])
test_x = x.values
#print(test_data.iloc[:, :5])#展示前5列

# 每个用户拿测试集预测——√
# Or中心服务器使用各个模型进行预测——×
# 合并最终预测结果，采用相对多数
final_pred =[]
# 这里的时间s4——两种方案——对应串行和并行
s4_start = datetime.datetime.now()
for u in range(num_user):
    print('collect {}th user\'s prediction result......'.format(u+1))
    predictions = workers[u].model.predict(test_x, num_iteration=workers[u].model.best_iteration)
    print('This is what we want:')
    print(type(predictions))
    y_pred = [list(x).index(max(x)) for x in predictions]
    print('result size:')
    print(sys.getsizeof(y_pred))
    if u == 0:
        final_pred = [y_pred]
    else:
        final_pred.append(y_pred)
print(type(final_pred))
s4_end = datetime.datetime.now()
s4 = (s4_end - s4_start).seconds
#print('S4 Time cost: ' + str(s4))

#print(final_pred)

pred_dataframe = DataFrame(final_pred)
final_result = pred_dataframe.mode()[0:1].values.tolist()
final_pred_result = final_result[0]
final_pred_result = [int(num) for num in final_pred_result]
print('final:-------------------------------')
print(final_pred_result)
#print(y.tolist())
print('The accuracy of prediction in test dataset is:' + str(accuracy_score(y, final_pred_result)))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==13:
           TP += 1
        if y_hat[i]==13 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]!=13:
           TN += 1
        if y_hat[i]!=13 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

tp, fp, tn, fn = perf_measure(y, final_pred_result)
# print(str(tp))
# print(str(fp))
# print(str(tn))
# print(str(fn))

fpr = fp / (tn + fp)
recall = tp / (tp + fn)
# precision = tp / (tp + fp)



reverse_attack_types = {v: k for k, v in ATTACK_TYPES.items()}
# 将编码映射回攻击类型名称
mapped_results = [reverse_attack_types[code] for code in final_pred_result]
# 打印结果
print(mapped_results)
df_mapped_results = pd.DataFrame(mapped_results, columns=['Attack Type'])
# 将DataFrame保存为CSV文件
df_mapped_results.to_csv('mapped_results.csv', index=False)
print("CSV文件已保存。")


print('The Recall score of prediction in test dataset is:' + str(recall))
if (tp + fp) != 0:
    precision = tp / (tp + fp)
    print('The Precision score of prediction in test dataset is:' + str(precision))
print('The False Positive Rate(FPR) of prediction in test dataset is:' + str(fpr))
# print('The micro F1_score of prediction in test dataset is:' + str(f1_score(y,final_pred_result, average='micro')))
print('The macro F1_score of prediction in test dataset is:' + str(f1_score(y,final_pred_result, average='macro')))

s1_end = datetime.datetime.now()

s1 = (s1_end - s1_start).seconds
#print('S1 Time cost: ' + str(s1))

tree_cost = s3 / num_user + s4 + (s1 - s2 -s3 -s4)
dataset_cost = s3 / num_user + s4 / num_user + (s1 - s2 -s3 -s4)
#print('Passing Tree: '+ str(tree_cost))
#print('Passing datasets: ' + str(dataset_cost))

'''
# 读取数据
dataset_path = '/Users/klaus_imac/klausHome/UESTC/数据竞赛/第一届协作学习与网络安全大赛/初赛/数据集/train/type-four-0-150000-samples.csv'
data = pd.read_csv(dataset_path)
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

y = np.array([
        ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
        for t in data.iloc[:, -1]
    ])
y = y.ravel()
print(y.shape)
x = data.drop(['Label'], axis=1).select_dtypes(exclude=['object'])
# print(type(x))
# y = pd.DataFrame(y)
# print(type(x.values))
# print(type(y.values))
# print(x.values.shape)
# print(y.values.shape)
train_x, test_x, train_y, test_y = train_test_split(x.values, y, test_size=0.2)

lgb_train = lgb.Dataset(train_x, train_y)
lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)

# 参数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',    # 设置提升类型
    'objective': 'multiclass',  # 目标函数
    'num_leaves': 40,           # 叶子节点数
    'learning_rate': 0.05,      # 学习速率
    'feature_fraction': 0.9,    # 建树的特征选择比例
    'bagging_fraction': 0.8,    # 建树的样本采样比例
    'bagging_freq': 5,          # k 意味着每 k 次迭代执行bagging
    'verbose': 1,                # =0 显示错误 (警告), >0 显示信息
    'num_class': 14,
}

model = lgb.train(params, lgb_train, num_boost_round=60, valid_sets=lgb_eval, early_stopping_rounds=5)
predictions = model.predict(test_x, num_iteration=model.best_iteration)
y_pred = [list(x).index(max(x)) for x in predictions]
print(y_pred)
print('The accuracy of prediction is:' + str(accuracy_score(test_y, y_pred)))

print('--------------------------------------')
# 模型存储
joblib.dump(model, 'loan_model.pkl')
# 模型加载
gbm = joblib.load('loan_model.pkl')
predictions = gbm.predict(test_x, num_iteration=model.best_iteration)
y_pred = [list(x).index(max(x)) for x in predictions]
print(y_pred)

# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
print(test_y.tolist())
print('The accuracy of prediction is:' + str(accuracy_score(test_y, y_pred)))
'''

'''
#iris数据集
# 加载数据
iris = datasets.load_iris()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

# 转换为Dataset数据格式
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)

# 参数
params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 10,
    'objective': 'multiclass',  # 目标函数
    'num_class': 3,
}

# 模型训练
gbm = lgb.train(params, train_data, valid_sets=[validation_data])

# 模型预测
y_pred = gbm.predict(X_test)
y_pred = [list(x).index(max(x)) for x in y_pred]
print(y_pred)

# 模型评估
print(accuracy_score(y_test, y_pred))
'''

'''
data_path = '/Users/klaus_imac/klausHome/UESTC/数据竞赛/第一届协作学习与网络安全大赛/初赛/数据集/train/type-four-0-150000-samples.csv'
data = pd.read_csv(data_path)
features_considered = ['Total Length of Fwd Packets', 'Fwd Packet Length Mean',
                               'Bwd Packet Length Min', 'Bwd Packet Length Std', 'Flow IAT Mean', 'Flow IAT Std',
                               'Flow IAT Min',
                               'Fwd IAT Min', 'Bwd IAT Mean', 'Fwd PSH Flags', 'Fwd Packets/s', 'Bwd Packets/s',
                               'SYN Flag Count',
                               'PSH Flag Count', 'ACK Flag Count', 'Average Packet Size', 'Subflow Fwd Bytes',
                               'Init_Win_bytes_forward',
                               'Init_Win_bytes_backward', 'Active Mean', 'Active Min']
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
x = data[features_considered]
y = np.array([
        ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
        for t in data.iloc[:, -1]
    ])
x = x.to_numpy().astype(np.float32)
train_data = (x, y)

def auc2(m, train, test):
    return (metrics.roc_auc_score(y,m.predict(train)),
                            metrics.roc_auc_score(y,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }

# grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
# grid_search.fit(x,y)
# grid_search.best_estimator_

d_train = lgb.Dataset(x, label=y)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300}
model2 = lgb.train(params, d_train)
auc2(model2, train_data, train_data)
'''



