from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def model_metrics(clf, X_train, X_test, y_train, y_test, pos_label=1):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    y_train_proba = clf.predict_proba(X_train)[:, pos_label]   
    y_test_proba = clf.predict_proba(X_test)[:, pos_label]
    
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1_score': f1_score,
        'auc': roc_auc_score
    }
    
    for metric_name, metric_func in metrics.items():
        try:
            if metric_name == 'accuracy':
                train_score = metric_func(y_train, y_train_pred)
                test_score = metric_func(y_test, y_test_pred)
            elif metric_name == 'auc':
                train_score = metric_func(y_train, y_train_proba)
                test_score = metric_func(y_test, y_test_proba)
            else:
                train_score = metric_func(y_train, y_train_pred, pos_label=pos_label)
                test_score = metric_func(y_test, y_test_pred, pos_label=pos_label)
            logging.info(f'[{metric_name}] 训练集：{train_score:.4f} 测试集：{test_score:.4f}')
        except ValueError as e:
            logging.warning(f'计算{metric_name}时出现错误：{e}')
    
    # ROC曲线
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, pos_label=pos_label)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba, pos_label=pos_label)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'Train - AUC:{auc(fpr_train, tpr_train):.4f}')
    plt.plot(fpr_test, tpr_test, label=f'Test - AUC:{auc(fpr_test, tpr_test):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# 使用示例
# model_metrics(clf, X_train, X_test, y_train, y_test)