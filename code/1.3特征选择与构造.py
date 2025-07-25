'''
本项目采用IV值过滤，再结合RF筛选。
筛选出IV值高于0.01的特征集合A,随机森林筛选出重要性前20的集合B.求A并B。
'''

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.先从保存好的 “数据清洗与缺失值处理之后的data.pkl” 中获取X
with open('tmp/数据清洗与缺失值处理之后的features.pkl','rb') as f:
    X = pickle.load(f, encoding = 'gbk')

data_orig = pd.read_csv("data/data.csv",encoding = 'gbk' )
y = data_orig.status


# 2.采用IV值过滤
def cal_iv(x, y, n_bins=6, null_value=np.nan,):
    # 剔除空值
    x = x[x != null_value]
    # 若 x 只有一个值，返回 0
    if len(x.unique()) == 1 or len(x) != len(y):
        return 0
    if x.dtype == np.number:
        # 数值型变量
        if x.nunique() > n_bins:
            # 若 nunique 大于箱数，进行分箱
            x = pd.qcut(x, q=n_bins, duplicates='drop')       
    # 计算IV
    groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
    t0, t1 = y.value_counts().index
    groups = groups / groups.sum()
    not_zero_index = (groups[t0] > 0) & (groups[t1] > 0)
    groups['iv_i'] = (groups[t0] - groups[t1]) * np.log(groups[t0] / groups[t1])
    iv = sum(groups['iv_i'])
    return iv
# 统计每个特征对应的 iv 值
fea_iv = X.apply(lambda x: cal_iv(x, y), axis=0).sort_values(ascending=False) 
#pd.apply()实现对于X的每一列计算其与标签列的iv值，然后合并成一个series,返回。
# 筛选 IV > 0.05 的特征
imp_fea_iv = fea_iv[fea_iv > 0.05].index
# 打印特征及其IV值
for feature in imp_fea_iv:
    iv_value = fea_iv[feature]
    print(f"Feature: {feature}, IV Value: {round(iv_value, 4)}")

# 可视化IV值大于0.05的特征
plt.figure(figsize=(12, 6))
bars = plt.bar(imp_fea_iv, fea_iv.loc[imp_fea_iv], color='skyblue')
plt.xlabel('Features')
plt.ylabel('IV Value')
plt.title('IV Value of Features > 0.05')

# 在柱状图上添加IV值标签，保留四位小数
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

# 横坐标的标签倾斜显示，调整旋转角度以便于阅读
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()



# 3.结合RF筛选
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
rf.fit(X, y)
rf_impc = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
# 筛选重要性前二十个特征
imp_fea_rf = rf_impc.index[:20]
top20_importances = rf_impc.values[:20]

# 创建柱状图
bars = plt.bar(imp_fea_rf, top20_importances, color='skyblue')

# 在柱状图上添加重要性值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 20 Important Features by Random Forest')
plt.xticks(rotation=90)  # 旋转横坐标标签，便于阅读
plt.tight_layout()
plt.show()
print(len(imp_fea_rf))
print(imp_fea_rf)


# 4.合并特征并筛选出有用特征
imp_fea = list(set(imp_fea_iv) | set(imp_fea_rf))
X_imp = X[imp_fea]
print(X_imp.shape)


# 训练看看效果
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt   
scaler = StandardScaler()
X_imp = scaler.fit_transform(X_imp)
X_train, X_test, y_train, y_test = train_test_split(X_imp,y,test_size=0.3, random_state=2018)
lr = LogisticRegression()
lr.fit(X_train, y_train)
from model_metrics import model_metrics
model_metrics(lr, X_train, X_test, y_train, y_test)
plt.show()


# 保存选择后的29个特征
import pickle
with open('tmp/特征选择与构造后的29个features.pkl','wb') as f:
    pickle.dump(X_imp,f)


print(y.shape)
with open('tmp/labels.pkl','wb') as f:
    pickle.dump(y,f)

X_imp.to_csv('tmp/特征选择与构造后的29个features.csv', index=False)
y.to_csv('tmp/labels.csv', index=False, header=['label'])