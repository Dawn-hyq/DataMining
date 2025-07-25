import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


# 1.导入数据
with open('tmp/特征选择与构造后的29个features.pkl','rb') as f:
    X = pickle.load(f, encoding = 'gbk')
with open('tmp/labels.pkl','rb') as f:
    y = pickle.load(f)


# 2.划分数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 2018)


import warnings
warnings.filterwarnings('ignore')

# 3.模型训练
from xgboost.sklearn import XGBClassifier
from model_metrics import model_metrics

#默认参数
xgb0 = XGBClassifier()
xgb0.fit(X_train, y_train)
model_metrics(xgb0, X_train, X_test, y_train, y_test)


# 4.调整参数
from sklearn.model_selection import GridSearchCV

# 定义可视化评估指标的函数
def plot_line_chart(cv_results, param_name, score_name, title, xlabel, ylabel):
    """
    绘制折线图
    :param cv_results: GridSearchCV 的 cv_results_ 属性
    :param param_name: 参数的名称
    :param score_name: 分数的名称
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    """
    param_values = cv_results[f'param_{param_name}'].data
    scores = cv_results[score_name]

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_heatmap(cv_results, param1_name, param2_name, score_name, title, xlabel, ylabel):
    """
    绘制热力图
    :param cv_results: GridSearchCV 的 cv_results_ 属性
    :param param1_name: 第一个参数的名称
    :param param2_name: 第二个参数的名称
    :param score_name: 分数的名称
    :param title: 图表标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    """
    param1_values = cv_results[f'param_{param1_name}'].data
    param2_values = cv_results[f'param_{param2_name}'].data
    scores = cv_results[score_name]

    unique_param1 = sorted(set(param1_values))
    unique_param2 = sorted(set(param2_values))

    data = np.zeros((len(unique_param1), len(unique_param2)))

    for i, param1 in enumerate(unique_param1):
        for j, param2 in enumerate(unique_param2):
            score = scores[(param1_values == param1) & (param2_values == param2)]
            if len(score) > 0:
                data[i, j] = score[0]

    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, fmt=".4f", cmap='viridis', linewidths=.5,
                xticklabels=unique_param2, yticklabels=unique_param1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 4.1首先从步长(learning rate)和迭代次数(n_estimators)入手。
param_test = {'n_estimators':range(20,200,20)}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=5, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, 
                                                  colsample_bytree=0.8, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
print(gsearch.best_params_, gsearch.best_score_)   
# {'n_estimators': 40} 0.793102401325268

# 绘制折线图
plot_line_chart(
    gsearch.cv_results_,
    param_name='n_estimators',
    score_name='mean_test_score',
    title='ROC-AUC vs. n_estimators',
    xlabel='n_estimators',
    ylabel='ROC-AUC'
)

# 4.2max_depth 和 min_child_weight 参数调优
param_test = {'max_depth':range(3,10,2), 'min_child_weight':range(1,12,2)}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=40, max_depth=5, 
                                                  min_child_weight=1, gamma=0, subsample=0.8, 
                                                  colsample_bytree=0.8, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
# gsearch.grid_scores_, 
print(gsearch.best_params_, gsearch.best_score_) 
# {'max_depth': 3, 'min_child_weight': 11} 0.7996100177171792

# 绘制热力图
plot_heatmap(
    gsearch.cv_results_,
    param1_name='max_depth',
    param2_name='min_child_weight',
    score_name='mean_test_score',
    title='ROC-AUC vs. max_depth and min_child_weight',
    xlabel='min_child_weight',
    ylabel='max_depth'
)

# 4.3gamma参数调优
param_test = {'gamma':[i/10 for i in range(0,6)]}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=40, max_depth=3, 
                                                  min_child_weight=11, gamma=0.0, subsample=0.8, 
                                                  colsample_bytree=0.8, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
# gsearch.grid_scores_, 
print(gsearch.best_params_, gsearch.best_score_) 
# {'gamma': 0.3} 0.7998020184851823

# 绘制折线图
plot_line_chart(
    gsearch.cv_results_,
    param_name='gamma',
    score_name='mean_test_score',
    title='ROC-AUC vs. gamma',
    xlabel='gamma',
    ylabel='ROC-AUC'
)

# 4.4调整subsample 和 colsample_bytree 参数
param_test = {'subsample':[i/10 for i in range(5,10)], 'colsample_bytree':[i/10 for i in range(5,10)]}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=40, max_depth=3, 
                                                  min_child_weight=11, gamma=0.3, subsample=0.8, 
                                                  colsample_bytree=0.8, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
# gsearch.grid_scores_, 
print(gsearch.best_params_, gsearch.best_score_) 
# {'colsample_bytree': 0.6, 'subsample': 0.6} 0.8031528800813998

# 绘制热力图
plot_heatmap(
    gsearch.cv_results_,
    param1_name='subsample',
    param2_name='colsample_bytree',
    score_name='mean_test_score',
    title='ROC-AUC vs. subsample and colsample_bytree',
    xlabel='colsample_bytree',
    ylabel='subsample'
)

# 4.5正则化参数调优
param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 0, 1, 100], 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1]}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=40, max_depth=3, 
                                                  min_child_weight=11, gamma=0.3, subsample=0.6, 
                                                  colsample_bytree=0.6, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
# gsearch.grid_scores_, 
print(gsearch.best_params_, gsearch.best_score_) 
# {'reg_alpha': 0.01, 'reg_lambda': 0.8} 0.8039588013773742

# 绘制热力图
plot_heatmap(
    gsearch.cv_results_,
    param1_name='reg_alpha',
    param2_name='reg_lambda',
    score_name='mean_test_score',
    title='ROC-AUC vs. reg_alpha and reg_lambda',
    xlabel='reg_lambda',
    ylabel='reg_alpha'
)

# 4.6回到第1）步，降低学习速率, 调整迭代次数(效果变坏不采用)
param_test = {'n_estimators':range(160,200,5)}
gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate =0.01, n_estimators=40, max_depth=3, 
                                                  min_child_weight=11, gamma=0.3, subsample=0.6, 
                                                  colsample_bytree=0.6, objective= 'binary:logistic', 
                                                  nthread=4,scale_pos_weight=3, seed=27), 
                        param_grid = param_test, scoring='roc_auc',n_jobs=1, cv=5)
gsearch.fit(X_train, y_train)
# gsearch.grid_scores_, 
print(gsearch.best_params_, gsearch.best_score_) 
# {'n_estimators': 195} 0.7970454725192395


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2018)

xgb = XGBClassifier(learning_rate =0.1, n_estimators=40, max_depth=3, 
                    min_child_weight=11, gamma=0.3, subsample=0.6, colsample_bytree=0.6, 
                    reg_alpha=0.01, reg_lambda=0.8,
                    objective= 'binary:logistic', 
                    nthread=4,scale_pos_weight=3, seed=27)
xgb.fit(X_train, y_train)
model_metrics(xgb, X_train, X_test, y_train, y_test)
