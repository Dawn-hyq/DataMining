import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 
import missingno as msno
import pickle
from sklearn.ensemble import RandomForestClassifier


# 1.整体分析
data_orig = pd.read_csv("data/data.csv",encoding = 'gbk' )
print(data_orig.shape)
print(data_orig.head())  #查看原始数据前5行
print(data_orig['status'].value_counts())  #查看标签列，0表示未逾期，1表示逾期
print(data_orig.describe())  #描述性统计分析

# 1.2 计算信息价值（IV）
def calculate_iv(df, feature, target):
    """
    计算单个特征的信息价值（IV）
    :param df: 数据框
    :param feature: 特征名称
    :param target: 目标变量名称
    :return: IV值
    """
    # 创建一个分组表，统计每个特征值的逾期和未逾期数量
    grouped = df.groupby(feature)[target].agg(['sum', 'count']).reset_index()
    grouped.columns = [feature, 'bad', 'total']
    grouped['good'] = grouped['total'] - grouped['bad']
    
    # 计算占比
    total_good = grouped['good'].sum()
    total_bad = grouped['bad'].sum()
    grouped['good_rate'] = grouped['good'] / total_good
    grouped['bad_rate'] = grouped['bad'] / total_bad
    
    # 避免除以0的情况
    grouped['good_rate'] = grouped['good_rate'].replace(0, np.finfo(float).eps)
    grouped['bad_rate'] = grouped['bad_rate'].replace(0, np.finfo(float).eps)
    
    # 计算WOE
    grouped['WOE'] = np.log(grouped['good_rate'] / grouped['bad_rate'])
    
    # 计算IV
    grouped['IV'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['WOE']
    iv_value = grouped['IV'].sum()
    
    return iv_value

# 遍历所有特征，计算IV值
target = 'status'
iv_results = {}
for col in data_orig.columns:
    if col != target:  # 跳过目标变量
        iv = calculate_iv(data_orig, col, target)
        iv_results[col] = iv

# 将IV结果转换为DataFrame并排序
iv_df = pd.DataFrame(list(iv_results.items()), columns=['Feature', 'IV'])
iv_df = iv_df.sort_values(by='IV', ascending=False)

# 筛选出IV大于0.05的前5个字段
selected_features = iv_df[iv_df['IV'] > 0.05].head(10)
print("IV大于0.05的字段：")
print(selected_features)

# 1.3 绘制Histogram 
# 选择需要绘制的特征
features = [
    'historical_trans_amount',
    'trans_amount_3_month',
    'pawns_auctions_trusts_consume_last_6_month',
    'repayment_capability',
    'consume_mini_time_last_1_month',
    'consfin_avg_limit'
]

# 绘制直方图和变化曲线
plt.figure(figsize=(20, 18))

# 遍历每个特征
for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)  # 每行2个子图，共3行
    sns.histplot(data_orig[feature], kde=True, bins=30, color='skyblue')
    plt.title(f'{feature} - Histogram with KDE')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# 调整布局
plt.tight_layout()
plt.show()

# 1.4 绘制散点图矩阵
target = 'status'
plt.figure(figsize=(20, 15))
sns.pairplot(data_orig[features + [target]], hue=target, markers=["o", "s"], palette="coolwarm")

# 调整布局
plt.tight_layout()
plt.show()

# 1.5 绘制热力图
# 计算相关性矩阵
corr_matrix = data_orig[features + [target]].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})
plt.xticks(rotation=30, ha='right')  # ha='right' 表示标签向右对齐
plt.title('Feature Correlation Heatmap')
plt.show()

# 2.缺失值可视化
# 2.1 计算每列的缺失值数量和缺失率
missing_data = data_orig.isnull().sum().reset_index()
missing_data.columns = ['Feature', 'Missing Count']
missing_data['Missing Rate'] = missing_data['Missing Count'] / len(data_orig)
# 筛选出有缺失值的列，并按缺失数量降序排序
missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)

# 2.2 绘制缺失值数量前十的特征的柱状图，并标注具体数量
top_missing_count = missing_data.head(10)
plt.figure(figsize=(14, 8))
sns.barplot(x='Missing Count', y='Feature', data=top_missing_count, color='skyblue')
# 在柱状图上标注具体数量
for i, value in enumerate(top_missing_count['Missing Count']):
    plt.text(value, i, f'{value}', va='center', ha='left', color='black', fontsize=10)
plt.title('Top 10 Features by Missing Value Count')
plt.xlabel('Missing Count')
plt.ylabel('Feature')
plt.show()

# 2.3 绘制缺失率前十的特征的扇形图
top_missing_rate = missing_data.sort_values(by='Missing Rate', ascending=False).head(10)
plt.figure(figsize=(8, 8))
top_missing_rate['Missing Rate'].plot.pie(
    startangle=140, 
    colors=plt.cm.Paired.colors, 
    labels=top_missing_rate['Feature'],  # 添加特征名称作为标签
    wedgeprops=dict(width=0.3),  # 设置环形扇形图的宽度
    autopct='%1.1f%%'  # 显示缺失率百分比
)
plt.title('Top 10 Features by Missing Rate')
plt.ylabel('')  # 隐藏 y 轴标签
plt.show()

# 2.4 绘制缺失值矩阵图
msno.matrix(data_orig)
plt.show()


# 3. 异常值可视化
# 先从保存好的 “数据清洗与缺失值处理之后的data.pkl” 中获取X
with open('tmp/数据清洗与缺失值处理之后的features.pkl','rb') as f:
    X = pickle.load(f, encoding = 'gbk')
y = data_orig.status

# 通过RF筛选出前8个重要的特征值
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X, y)
rf_impc = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("特征重要性指标：")
print(rf_impc.head(8))
imp_fea_rf = rf_impc.index[:8]

# 创建一个包含8个子图的图形，每个特征一个坐标轴
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 20))  # 4行2列
axes = axes.flatten()  # 将axes数组扁平化，方便迭代

# 绘制重要性前8个特征的箱线图
for i, feature in enumerate(imp_fea_rf):
    sns.boxplot(data=data_orig[[feature]], ax=axes[i])
    axes[i].set_title(f'Box Plot for {feature}')
    axes[i].set_ylabel('Value')
    axes[i].set_xlabel('')
plt.show()

