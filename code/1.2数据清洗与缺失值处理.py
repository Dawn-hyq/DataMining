import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns  # 基于Matplotlib的Python数据可视化库


# 1.读取原始数据 并 进行数据探索
data_orig = pd.read_csv("data/data.csv",encoding = 'gbk' )
print(data_orig.head())  #查看原始数据前5行
print(data_orig['status'].value_counts())  #查看标签列，0表示未逾期，1表示逾期
print(data_orig.describe())  #描述性统计分析

def overall(data):  #观察并打印特征的数据类型分布
    typedic= {} # 类型字典
    for name in data.columns:
        typedic[str(data[name].dtype)] = typedic.get(str(data[name].dtype),[])+[name]
    for key,value in typedic.items():
        print('我们有 {} 列是 {} 类型, 他们是 {}\n'.format(len(value),key,value))
print(overall(data_orig))


# 2.数据预处理
# 2.1删除无关特征（单一值列、无意义特征）
data = data_orig.copy()

# 2.1.1 删除单一值列（删掉了两个特征：'source' 'bank_card_no'）
def same_value_delete(data):
    for name in data.columns:
        if len(data[name].value_counts())==1:
            data.drop(name,axis = 1,inplace = True)
    return data
data = same_value_delete(data)
print(data.shape)
print(overall(data))

# 2.1.2 删除无意义特征（删掉了四个特征：'Unnamed: 0' 'custid' 'trade_no' 'id_name'）
data.iloc[:,0].hist()
sns.countplot(data = data, x = 'trans_fail_top_count_enum_last_1_month', hue = 'status')
sns.countplot(data = data, x = 'Unnamed: 0', hue = 'status')
plt.plot()
plt.show()
print(len(set(data_orig.iloc[:,0])), len(data['custid'].value_counts()), len(data['trade_no'].value_counts()), len(data['id_name'].value_counts()))
data.drop(columns=['Unnamed: 0','custid','trade_no','id_name'],inplace=True)
print(data.shape)

# 2.2类型转换
print(len(data['reg_preference_for_trad'].value_counts()))
print(data['reg_preference_for_trad'].value_counts())
data['reg_preference_for_trad'].replace({'一线城市':5,'二线城市':4,'三线城市':3,'境外':2,'其他城市':1},inplace = True)

# 2.3缺失值处理
print(data.info())
# 2.3.1删除缺失率大的特征
print(data['student_feature'].value_counts(dropna=False))
sns.countplot(data=data, x='student_feature', hue='status')
# 获取每个条形图的位置和高度
bars = plt.gca().patches
x_labels = data['student_feature'].unique()
y_labels = data['status'].unique()
# 在每个条形图上添加具体数据标注
for bar in bars:
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    plt.text(x, y, f'{int(y)}', ha='center', va='bottom')
plt.show()
print(data[data['student_feature']==2.0]['status'])
del(data['student_feature'])
# print(data.shape)

# 2.3.2填补缺失率小的特征
# （1）时间特征处理
y = data.status
X = data.drop('status', axis=1)
dateFeatures = ['first_transaction_time', 'latest_query_time', 'loans_latest_time']
X_date = data[dateFeatures]
print(X_date.head())
import warnings
warnings.filterwarnings('ignore')
# 首先用中位数填充 first_transaction_time 缺失值
X_date['first_transaction_time'].fillna(X_date['first_transaction_time'].median(), inplace = True)
# 转成字符串型日期
X_date['first_transaction_time'] = X_date['first_transaction_time'].apply(lambda x:str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8])
print(X_date.head())
# 特征衍生：年份、月份、星期几
X_date['first_transaction_time_year'] = pd.to_datetime(X_date['first_transaction_time']).dt.year
X_date['first_transaction_time_month'] = pd.to_datetime(X_date['first_transaction_time']).dt.month
X_date['first_transaction_time_weekday'] = pd.to_datetime(X_date['first_transaction_time']).dt.weekday

X_date['latest_query_time_year'] = pd.to_datetime(X_date['latest_query_time']).dt.year
X_date['latest_query_time_month'] = pd.to_datetime(X_date['latest_query_time']).dt.month
X_date['latest_query_time_weekday'] = pd.to_datetime(X_date['latest_query_time']).dt.weekday

X_date['loans_latest_time_year'] = pd.to_datetime(X_date['loans_latest_time']).dt.year
X_date['loans_latest_time_month'] = pd.to_datetime(X_date['loans_latest_time']).dt.month
X_date['loans_latest_time_weekday'] = pd.to_datetime(X_date['loans_latest_time']).dt.weekday

# 填充缺失值
X_date['latest_query_time_year'].fillna(X_date['latest_query_time_year'].median(), inplace = True)
X_date['latest_query_time_month'].fillna(X_date['latest_query_time_month'].median(), inplace = True)
X_date['latest_query_time_weekday'].fillna(X_date['latest_query_time_weekday'].median(), inplace = True)

X_date['loans_latest_time_year'].fillna(X_date['loans_latest_time_year'].median(), inplace = True)
X_date['loans_latest_time_month'].fillna(X_date['loans_latest_time_month'].median(), inplace = True)
X_date['loans_latest_time_weekday'].fillna(X_date['loans_latest_time_weekday'].median(), inplace = True)

#删除原来的dateFeatures
X_date.drop(dateFeatures, axis = 1, inplace=True)

# （2）类别特征处理
# 筛选出类别型特征
for col in X:
    cnt = len(X[col].unique())
    if cnt < 15:
        print(col, cnt, X[col].unique())

categoryFeatures = ['regional_mobility', 'is_high_user', 'avg_consume_less_12_valid_month', 
                    'top_trans_count_last_1_month', 'reg_preference_for_trad', 'railway_consume_count_last_12_month',
                    'jewelry_consume_count_last_6_month']
X_cate = X[categoryFeatures]

print(X_cate.describe().T\
    .assign(missing_pct=data.apply(lambda x : (len(x)-x.count())/len(x))))  #.assign()添加了缺失率属性

#用众数填充类别的缺失值
for col in X_cate.columns:
    summ = X_cate[col].isnull().sum()
    if summ:
        X_cate[col].fillna(X_cate[col].mode()[0], inplace = True)
print(X_cate.isnull().any().sum()) #统计存在缺失值的列的个数

# （3）数值型特征处理
# 筛选出数值型特征
X_num = X.select_dtypes(exclude=['O']).copy()    # 不是复制视图, 所以加copy()
print(X_num.shape)
for col in X_num.columns:
    if col in dateFeatures + categoryFeatures:
        print(col)
        X_num.drop(col, axis = 1, inplace = True)
print(X_num.shape)

print((X_num.describe().T
 .drop(['25%','50%','75%'],axis=1)  #缺失情况描述，去掉25%，50%，75%的情况
 .assign(missing_pct=data.apply(lambda x: (len(x)-x.count())/len(x)))).T)

# 统计各列缺失值的比例
col_missing = {}
for col in X_num.columns:
    summ = X_num[col].isnull().sum()
    if summ:
        col_missing[col] = float('%.4f'%(summ*100/len(data)))
col_missing = sorted(col_missing.items(), key = lambda d:-d[1])
for col, rate in col_missing[:10]:
    print(rate, '%', col)

# 缺失特征用中位数填充。
for col in X_num.columns:
    summ = X_num[col].isnull().sum()
    if summ:
        X_num[col].fillna(X_num[col].median(), inplace = True)

print(X_num.isnull().any().sum())


# 2.4特征拼接与存储
X = pd.concat([X_cate, X_date, X_num], axis=1)
# print(X.shape)

import pickle
with open('tmp/数据清洗与缺失值处理之后的features.pkl', 'wb') as f:
    pickle.dump(X, f)


# 2.5异常值处理
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像时负号'-'显示为方块的问题

# 加载数据
with open('tmp/数据清洗与缺失值处理之后的features.pkl', 'rb') as f:
    X = pickle.load(f, encoding='gbk')

# 获取特征名称
features = X.columns

# 定义量级范围和对应的特征列表
thresholds = [2.5, 20, 40, 100, 200, 1000, 2500, 10000, 25000, 300000, float('inf')]
feature_lists = [[] for _ in range(len(thresholds))]
titles = [
    '量级 < 2.5', '2.5 ≤ 量级 < 20', '20 ≤ 量级 < 40', '40 ≤ 量级 < 100',
    '100 ≤ 量级 < 200', '200 ≤ 量级 < 1000', '1000 ≤ 量级 < 2500',
    '2500 ≤ 量级 < 10000', '10000 ≤ 量级 < 25000', '25000 ≤ 量级 < 300000', '量级 ≥ 300000'
]

# 根据量级将特征分类
for feature in features:
    max_value = X[feature].max()
    for i in range(len(thresholds)):
        if max_value < thresholds[i]:
            feature_lists[i].append(feature)
            break

# 定义一个函数，用于绘制箱线图
def plot_boxplot(feature_lists, titles, fig_title):
    num_plots = len(feature_lists)
    fig, axes = plt.subplots(1, num_plots, figsize=(20, 6))  # 创建一行多列的布局
    if num_plots == 1:
        axes = [axes]  # 如果只有一个子图，将 axes 转换为列表
    for i, features in enumerate(feature_lists):
        if not features:  # 如果特征列表为空
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].text(0.5, 0.5, '无特征', fontsize=12, ha='center', va='center')  # 在子图中显示“无特征”
            axes[i].axis('off')  # 关闭坐标轴
        else:
            sns.boxplot(data=X[features], ax=axes[i])
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90)  # 旋转x轴标签以便显示
    plt.suptitle(fig_title, fontsize=16)  # 添加总标题
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，为总标题留出空间
    plt.show()

# 绘制箱线图
plot_boxplot(feature_lists[:3], titles[:3], '低量级特征箱线图')
plot_boxplot(feature_lists[3:6], titles[3:6], '中低量级特征箱线图')
plot_boxplot(feature_lists[6:9], titles[6:9], '中高量级特征箱线图')
plot_boxplot(feature_lists[9:], titles[9:], '高量级特征箱线图')