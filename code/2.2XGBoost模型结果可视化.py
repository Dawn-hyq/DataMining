import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据
with open('tmp/特征选择与构造后的29个features.pkl','rb') as f:
    X = pickle.load(f, encoding = 'gbk')
with open('tmp/labels.pkl','rb') as f:
    y = pickle.load(f)

# 显示特征名称
feature_names = X.columns.tolist()

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2018)

# 默认参数模型
xgb0 = XGBClassifier()
xgb0.fit(X_train, y_train)
y_train_proba0 = xgb0.predict_proba(X_train)[:,1]
y_test_proba0 = xgb0.predict_proba(X_test)[:,1]

# 调参后的模型
xgb = XGBClassifier(learning_rate=0.1, n_estimators=40, max_depth=3, 
                   min_child_weight=11, gamma=0.3, subsample=0.6, colsample_bytree=0.6, 
                   reg_alpha=0.01, reg_lambda=0.8,
                   objective='binary:logistic', 
                   nthread=4, scale_pos_weight=3, seed=27)
xgb.fit(X_train, y_train)
y_train_proba = xgb.predict_proba(X_train)[:,1]
y_test_proba = xgb.predict_proba(X_test)[:,1]

# 计算ROC曲线和AUC值
fpr_train0, tpr_train0, _ = roc_curve(y_train, y_train_proba0)
fpr_test0, tpr_test0, _ = roc_curve(y_test, y_test_proba0)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

roc_auc_train0 = auc(fpr_train0, tpr_train0)
roc_auc_test0 = auc(fpr_test0, tpr_test0)
roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test = auc(fpr_test, tpr_test)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_train0, tpr_train0, label=f'Default Train - AUC:{roc_auc_train0:.4f}')
plt.plot(fpr_test0, tpr_test0, label=f'Default Test - AUC:{roc_auc_test0:.4f}')
plt.plot(fpr_train, tpr_train, label=f'Tuned Train - AUC:{roc_auc_train:.4f}')
plt.plot(fpr_test, tpr_test, label=f'Tuned Test - AUC:{roc_auc_test:.4f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Default and Tuned XGBoost Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 绘制特征重要性直方图
def plot_top_n_feature_importances(model, feature_names, n=20, title='Top N Feature Importances'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 从高到低排序
    top_n_indices = indices[:n]  # 只取前n个特征

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(top_n_indices)), importances[top_n_indices], color='lightblue', align='center')
    plt.xticks(range(len(top_n_indices)), [feature_names[i] for i in top_n_indices], rotation=90)
    plt.xlim([-1, len(top_n_indices)])
    plt.tight_layout()
    plt.show()

plot_top_n_feature_importances(xgb, feature_names, n=20, title='Top 20 Feature Importances')