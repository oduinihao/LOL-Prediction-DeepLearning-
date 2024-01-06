import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


h_batch_size = 10000
h_hidden_size = [4096, 2048, 1024]


def main():
    lol_data = read_data()
    lol_data = normalize_data(lol_data)
    train_svm_classifier(lol_data)


def train_svm_classifier(np_data):
    # 分离特征和标签
    X = np_data[:, :-1]  # 所有行，除了最后一列
    y = np_data[:, -1]   # 所有行，只取最后一列

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE根据训练集生成样本
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    sampling_strategy = {cls: int(count / total_samples * 10000) for cls, count in
                         zip(unique_classes, class_counts)}
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=1)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 创建 SVM 模型，使用线性核
    svm_model = SVC(kernel='linear', C=1000000.0)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_trained = svm_model.predict(X_train)
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    train_accuracy = accuracy_score(y_trained, y_train)
    print(f"Train Accuracy: {train_accuracy:.2f}")
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {test_accuracy:.2f}")

    return svm_model


def read_data():
    file_path = 'high_diamond_ranked_10min.csv'
    data = pd.read_csv(file_path)

    # 删除 'gameId' 特征
    data.drop('gameId', axis=1, inplace=True)

    # 将 'blueWins' 放置在最后
    blue_wins = data.pop('blueWins')
    data['blueWins'] = blue_wins

    data = data.to_numpy()

    return data


def normalize_data(data):

    # 计算每列的最小值和最大值
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # 避免除以零，将最大值和最小值相同的列的最大值稍微增加一点
    max_vals[max_vals == min_vals] += 0.0001

    # 归一化数据
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


if __name__ == '__main__':
    main()

