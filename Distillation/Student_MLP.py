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


h_batch_size = 1000000
h_hidden_size = [128, 512, 512]
h_weight_decay = 0
h_dropout_rate = [0, 0.3, 0.5]


def main():
    train_loader, test_loader = create_dataloaders("svm_train_data.csv", "svm_test_data.csv", h_batch_size)

    model = MLP()

    train_model(model, train_loader, test_loader, epochs=1000000)


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.LeakyReLU()

        # 适配不同大小的输入和输出，以便于残差连接
        self.adjust_dimensions = (input_size != output_size)
        if self.adjust_dimensions:
            self.dimension_adapter = nn.Linear(input_size, output_size)

    def forward(self, x):
        identity = x

        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # 如果输入和输出尺寸不同，调整尺寸以匹配
        if self.adjust_dimensions:
            identity = self.dimension_adapter(identity)

        # 添加残差连接
        out += identity
        return out


class MLP(nn.Module):
    def __init__(self, input_size=38, hidden_sizes=h_hidden_size, output_size=1, dropout_rate=h_dropout_rate):
        super(MLP, self).__init__()
        layers = []

        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(ResidualBlock(prev_size, hidden_size, dropout_rate[i]))
            prev_size = hidden_size

        # 构建输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())  # Sigmoid 激活函数

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def load_data(file_path):
    # 加载数据
    data = np.loadtxt(file_path, delimiter=",")

    # 分离特征和标签
    X = data[:, :-2]  # 所有行，除了最后两列
    y = data[:, -2]  # 所有行，只取倒数第二列（真实标签）
    y_svm_prob = data[:, -1]  # 所有行，只取最后一列（SVM的概率标签）

    # 转换为PyTorch张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_svm_prob_tensor = torch.tensor(y_svm_prob, dtype=torch.float32)

    return X_tensor, y_tensor, y_svm_prob_tensor


def create_dataloaders(train_file_path, test_file_path, batch_size):
    # 加载训练和测试数据
    X_train, y_train, y_train_svm_prob = load_data(train_file_path)
    X_test, y_test, y_test_svm_prob = load_data(test_file_path)

    # 创建TensorDataset
    train_dataset = TensorDataset(X_train, y_train, y_train_svm_prob)
    test_dataset = TensorDataset(X_test, y_test, y_test_svm_prob)

    # 创建DataLoader
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


def train_model(model, train_loader, test_loader, epochs=10, device=torch.device("cuda:0"), soft_target_weight=0.7):
    model.to(device)
    criterion_hard = nn.BCELoss()  # 硬目标损失
    criterion_soft = nn.BCELoss()  # 软目标损失
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=h_weight_decay)

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
        for data, target, soft_target in train_loader_tqdm:  # 加载软目标
            data, target, soft_target = data.to(device), target.to(device), soft_target.to(device)

            optimizer.zero_grad()
            output = model(data)

            # 计算加权损失
            loss_hard = criterion_hard(output, target.unsqueeze(1))
            loss_soft = criterion_soft(output, soft_target.unsqueeze(1))
            loss = soft_target_weight * loss_soft + (1 - soft_target_weight) * loss_hard  # 加权组合

            loss.backward()
            optimizer.step()

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
            for data, target, _ in test_loader_tqdm:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion_hard(output, target.unsqueeze(1))
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
            torch.save(best_model, 'MLP_model.pth')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

        # 实时更新损失图和准确率图
        plot_loss(training_loss, testing_loss, ax_loss)
        plot_accuracy(train_accuracies, test_accuracies, ax_acc)

    plt.close(fig)


if __name__ == '__main__':
    main()

