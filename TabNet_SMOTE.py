import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score


h_batch_size = 1000000
h_hidden_size = [512, 512]


def main():
    # 读取和处理数据
    X, y = read_and_process_data()

    # 创建带有 SMOTE 的 DataLoader
    X_train, y_train, X_test, y_test = create_data_with_smote_for_tabnet(X, y)
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-1),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax'  # "sparsemax", "entmax" 或 None
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['accuracy'],
        max_epochs=500,
        patience=20,
        batch_size=16384,
        virtual_batch_size=1024,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    # 进行预测
    predictions = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy}")


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


def read_and_process_data():
    # 读取并处理数据
    data = read_data()
    data = normalize_data(data)

    # 分离特征和标签
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def create_data_with_smote_for_tabnet(X, y, train_test_split_ratio=0.8):
    # 分割原始数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_test_split_ratio))

    # 应用 SMOTE 只在训练数据上
    target_sample_count = 100000
    class_counts = np.unique(y_train, return_counts=True)[1]
    sampling_strategy = {class_id: target_sample_count for class_id, count in enumerate(class_counts)}

    # 应用 SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=2)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, y_train_res, X_test, y_test


def create_dataloaders(np_data, batch_size=h_batch_size, train_test_split=0.8):

    # 分离输入和输出
    X = np_data[:, :-1]
    y = np_data[:, -1]

    # 将 NumPy 数组转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 计算训练数据和测试数据的大小
    total_samples = len(np_data)
    train_size = int(total_samples * train_test_split)
    test_size = total_samples - train_size

    # 划分数据集
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_loss(training_loss, testing_loss, ax_loss):
    ax_loss.clear()
    ax_loss.plot(training_loss, label='Training Loss')
    ax_loss.plot(testing_loss, label='Testing Loss')
    ax_loss.set_title("Training and Testing Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    plt.draw()
    plt.pause(0.01)


def plot_accuracy(train_accuracies, test_accuracies, ax_acc):
    ax_acc.clear()
    ax_acc.plot(train_accuracies, label='Train Accuracy')
    ax_acc.plot(test_accuracies, label='Test Accuracy')
    ax_acc.set_title("Training and Testing Accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    plt.draw()
    plt.pause(0.001)


def hook_fn(module, input, output):
    # 计算输出中非零元素的比例
    non_zeros = torch.count_nonzero(output)
    total_elements = output.numel()
    print(f"{module.__class__.__name__} - 非零输出比例: {non_zeros.float() / total_elements:.2f}")


def train_model(model, train_loader, test_loader, epochs=10, device=torch.device("cuda:0")):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 准备绘图
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    training_loss = []
    testing_loss = []
    train_accuracies = []
    test_accuracies = []

    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练过程
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch")
        for data, target in train_loader_tqdm:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清除梯度
            output = model(data)  # 前向传播
            loss = criterion(output, target.unsqueeze(1))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            train_loss += loss.item()
            predicted = output.round()
            train_correct += (predicted == target.unsqueeze(1)).sum().item()
            train_total += target.size(0)

        train_loss /= train_total
        training_loss.append(train_loss)
        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)

        # 测试过程
        model.eval()  # 设置模型为评估模式
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", unit="batch")
        with torch.no_grad():
            for data, target in test_loader_tqdm:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.unsqueeze(1))
                test_loss += loss.item()
                predicted = output.round()
                test_correct += (predicted == target.unsqueeze(1)).sum().item()
                test_total += target.size(0)
                test_loader_tqdm.set_postfix({'Accuracy': 100 * test_correct / test_total, 'best_epoch': best_epoch,
                                              'best_accuracy': best_accuracy})

        test_loss /= test_total
        testing_loss.append(test_loss)
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        if test_loss < best_test_loss:
            best_epoch = epoch + 1
            best_test_loss = test_loss
            best_model = model.state_dict()  # 保存模型参数
            torch.save(best_model,
                       'E:\File\SyncFile\BaiduSyncdisk\Graduate stage\课程作业\深度学习\model\MLP_model.pth')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        # 实时更新损失图和准确率图
        plot_loss(training_loss, testing_loss, ax_loss)
        plot_accuracy(train_accuracies, test_accuracies, ax_acc)

    plt.close(fig)


if __name__ == '__main__':
    main()

