import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from Triage_station import train_nurse_kmeans, classify_patients
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


h_batch_size = 1000000
h_epochs = 1000
h_learn_rate = 5e-5
h_weight_decay = 1e-4
h_hidden_size = [512, 512]
h_dropout_rate = [0.3, 0.3]
h_cluster_with_label = True
h_save_path = "E:\File\SyncFile\BaiduSyncdisk\Graduate stage\课程作业\深度学习\Hospital\doctor"


def main():
    _, result = establish_hospital(1, 10000, 0.0,
                                   doctor_model="SVM")
    _, result = establish_hospital(4, 10000, 0.75,
                                   result_of_pre_hospital=result,
                                   cluster_with_label=True,
                                   train_with_label=True)
    _, result = establish_hospital(4, 10000, 0.75,
                                   result_of_pre_hospital=result,
                                   cluster_with_label=True,
                                   train_with_label=True)
    _, result = establish_hospital(5, 10000, 0.75,
                                   result_of_pre_hospital=result,
                                   cluster_with_label=True,
                                   train_with_label=True)
    return 0


class MLP(nn.Module):
    def __init__(self,
                 hidden_sizes=h_hidden_size,
                 dropout_rate=h_dropout_rate,
                 input_size=40,
                 output_size=1,):
        super(MLP, self).__init__()

        self.input_size = input_size

        # 构建隐藏层
        layers = []
        for i in range(len(hidden_sizes)):
            # 添加线性层
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

            # 添加层归一化
            layers.append(nn.LayerNorm(hidden_sizes[i]))

            # 添加激活函数
            layers.append(nn.LeakyReLU())

            # 添加dropout
            layers.append(nn.Dropout(dropout_rate[i]))

        # 构建输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Sigmoid())  # Sigmoid 激活函数

        # 将所有层合并为一个序列
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


def svm_classification(data_tra, data_tes):
    # 分离训练数据的特征和标签
    X_tra = data_tra[:, :-1]
    Y_tra = data_tra[:, -1]

    # 分离测试数据的特征和标签
    X_tes = data_tes[:, :-1]
    Y_tes = data_tes[:, -1]

    # 创建 SVM 分类器，启用概率估计
    svm_classifier = SVC(kernel='rbf', probability=True)
    svm_classifier.fit(X_tra, Y_tra)  # 训练模型

    # 在测试数据上进行预测概率
    Y_result = svm_classifier.predict_proba(X_tes)
    Y_result = Y_result[:, -1]
    print("类别顺序:", svm_classifier.classes_)

    # 计算并打印准确率
    Y_pred = svm_classifier.predict(X_tes)  # 仍然进行一次常规预测，以便计算准确率
    accuracy = accuracy_score(Y_tes, Y_pred)
    print(f"SVM分类器在测试集上的准确率: {accuracy:.2f}")

    return Y_result


def establish_hospital(n_doctors,
                       n_samples,
                       doctors_standard,
                       result_of_pre_hospital=None,
                       cluster_with_label=False,
                       train_with_label=False,
                       doctor_model="MLP"):
    # 读取并处理数据
    data_total = normalize_data(read_data())
    data_tra = data_total[:int(len(data_total) * 0.8)]
    data_tes = data_total[int(len(data_total) * 0.8):]

    if doctor_model == "SVM":
        diagnosis_strategy = [4]
        total_result = svm_classification(data_tra, data_tes)
        return diagnosis_strategy, total_result

    data_tra, nurse, diagnose_rate_1 = train_nurse_kmeans(n_doctors,
                                                          data_tra,
                                                          cluster_with_label=cluster_with_label)
    print("\n")
    data_tes, diagnose_rate_2 = classify_patients(data_tes,
                                                  nurse,
                                                  result_of_pre_hospital=result_of_pre_hospital,
                                                  cluster_with_label=cluster_with_label)

    # 初始化用于存储信息的变量
    final_train_accuracy = [0] * n_doctors
    hospital_accuracy = 0
    diagnosis_strategy = [0] * n_doctors
    accuracy_1 = [0] * n_doctors
    accuracy_2 = [0] * n_doctors
    total_result = []

    # 检查分类器在训练集上的分类表现
    for i in range(0, n_doctors):
        if diagnose_rate_1[i] - diagnose_rate_2[i] > 0.1:
            print(f"第{i}簇训练集与测试集分类差距大于0.1,分类器失效！")

        # 针对每个簇分配不同策略
    for i in range(0, n_doctors):
        diagnosis_strategy[i] = 1  # 默认输出1为蓝色方赢，0为蓝色方输
        # 将蓝色方胜率转化为准确率，修正默认输出
        if diagnose_rate_1[i] < 0.5:
            accuracy_1[i] = 1 - diagnose_rate_1[i]
            diagnosis_strategy[i] = 0
        else:
            accuracy_1[i] = diagnose_rate_1[i]

        if diagnose_rate_2[i] < 0.5:
            accuracy_2[i] = 1 - diagnose_rate_2[i]
            diagnosis_strategy[i] = 0
        else:
            accuracy_2[i] = diagnose_rate_2[i]

        # 从上层结果中提取出对应簇的结果
        if train_with_label:
            result_of_pre_hospital = data_tes[i][:, -1]
            data_tes[i] = data_tes[i][:, :-1]

        if accuracy_1[i] > doctors_standard:  # 判断是否无需引入专家网络
            print(f"第 {i + 1} 簇的准确率为 {accuracy_1[i]:.2f} 已经满足您的要求，无需训练专家网络！\n\n")
            final_train_accuracy[i] = accuracy_2[i] * 100

        else:  # 准确率低于要求，引入专家网络
    
            print(f"第 {i + 1} 簇的准确率为 {accuracy_1[i]:.2f} 不满足您的要求，开始训练专家网络！")
            final_train_accuracy[i], doctor_prediction = train_doctor(data_tra[i],
                                                                      data_tes[i],
                                                                      i,
                                                                      n_doctors,
                                                                      n_samples,
                                                                      nurse,
                                                                      diagnose_rate_1[i],
                                                                      result_of_pre_hospital,
                                                                      train_with_label)

            if final_train_accuracy[i] > accuracy_1[i] * 100:  # 专家网络成功提升准确率
                print(f"第 {i+ 1} 簇专家模型训练准确度为 {final_train_accuracy[i]:.2f} ，超过无专家网络准确率，录用！\n\n")
                diagnosis_strategy[i] = 3  # 3为使用专家网络标识
            else:  # 专家网络未能提升准确率
                print(f"第 {i + 1} 簇专家模型训练准确度为 {final_train_accuracy[i]:.2f}，未超过无专家准确率，不使用专家网络!\n\n")
                final_train_accuracy[i] = accuracy_2[i] * 100

        if diagnosis_strategy[i] == 3:
            total_result.append(doctor_prediction)  # 将当前簇的预测结果添加到总结果列表中
        else:
            default_prediction = np.full(len(data_tes[i]), diagnosis_strategy[i])
            total_result.append(default_prediction)

        hospital_accuracy += final_train_accuracy[i] * len(data_tes[i])

    hospital_accuracy = hospital_accuracy / (len(data_total) * 0.2)
    print(f"每个簇的准确率为{final_train_accuracy}")
    print(f"模型在训练集上的总准确率为{hospital_accuracy}")

    # 将所有簇的预测结果合并为一个数组
    total_result = np.concatenate(total_result)

    return diagnosis_strategy, total_result


def train_doctor(data_tra, data_tes, i, n_doctors, n_samples, nurse, diagnose_rate,
                 result_of_pre_hospital=None, train_with_label=False):

    tra_loader, tes_loader = smote_dataloader(data_tra, data_tes, n_samples, i, n_doctors, nurse,
                                              diagnose_rate, train_with_label, result_of_pre_hospital)
    model = MLP()
    accuracy, result = case_study(model, tra_loader, tes_loader, i, n_doctors,
                                  train_with_label=train_with_label,
                                  result_of_pre_hospital=result_of_pre_hospital)

    return accuracy, result


def smote_dataloader(data_tra, data_tes, n_samples, i, n_doctors, nurse, diagnose_rate,
                     train_with_label, result_of_pre_hospital, batch_size=h_batch_size):

    X_tra = data_tra[:, :-1]
    Y_tra = data_tra[:, -1]
    X_tes = data_tes[:, :-1]
    Y_tes = data_tes[:, -1]

    # SMOTE根据训练集生成样本
    unique_classes, class_counts = np.unique(Y_tra, return_counts=True)
    total_samples = len(Y_tra)
    sampling_strategy = {cls: int(count / total_samples * n_samples) for cls, count in
                         zip(unique_classes, class_counts)}
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=1)
    X_tra, Y_tra = smote.fit_resample(X_tra, Y_tra)

    # 检查样本质量
    # check_generated_sample_quality(X_tra, Y_tra, i, n_doctors, nurse)

    # 将 NumPy 数组转换为 PyTorch 张量
    X_tra_tensor = torch.tensor(X_tra, dtype=torch.float32)
    Y_tra_tensor = torch.tensor(Y_tra, dtype=torch.float32)
    X_tes_tensor = torch.tensor(X_tes, dtype=torch.float32)
    Y_tes_tensor = torch.tensor(Y_tes, dtype=torch.float32)

    # 创建 TensorDataset
    tra_dataset = TensorDataset(X_tra_tensor, Y_tra_tensor)
    tes_dataset = TensorDataset(X_tes_tensor, Y_tes_tensor)

    # 创建 DataLoader
    tra_loader = DataLoader(tra_dataset, batch_size=batch_size, shuffle=False)
    tes_loader = DataLoader(tes_dataset, batch_size=batch_size, shuffle=False)

    return tra_loader, tes_loader


def case_study(model, tra_loader, tes_loader, i, n_doctor, learn_rate=h_learn_rate, weight_decay=h_weight_decay,
               save_path=h_save_path, epochs=h_epochs, device=torch.device("cuda:0"),
               result_of_pre_hospital=None, train_with_label=False):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    # 初始化准备绘图
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    total_train_loss = []
    total_test_loss = []
    total_train_accuracy = []
    total_test_accuracy = []

    # 初始化准备训练与测试
    best_test_loss = float('inf')
    best_model = None
    best_epoch = 0
    best_train_accuracy = 0
    best_test_accuracy = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_counts = 0

        # 训练过程
        train_loader_tqdm = tqdm(tra_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch")
        for inputs, target in train_loader_tqdm:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = output.round()
            train_correct += (predicted == target.unsqueeze(1)).sum().item()
            train_counts += target.size(0)
            train_loader_tqdm.set_postfix({'训练损失:': train_loss / train_counts})

        train_loss /= train_counts
        total_train_loss.append(train_loss)
        train_accuracy = 100 * train_correct / train_counts
        total_train_accuracy.append(train_accuracy)

        # 测试过程
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        test_loader_tqdm = tqdm(tes_loader, desc=f"Epoch {epoch + 1}/{epochs} [Test]", unit="batch")
        with torch.no_grad():
            for inputs, target in test_loader_tqdm:
                inputs, target = inputs.to(device), target.to(device)
                output = model(inputs)

                if train_with_label:
                    # 将模型输出转换为CPU NumPy数组以进行加权求和
                    output_numpy = output.cpu().numpy()
                    result_of_pre_hospital = result_of_pre_hospital.reshape(-1, 1)
                    output_numpy = output_numpy * 0.5 + result_of_pre_hospital * 0.5
                    # 转换回PyTorch张量以计算损失和准确率
                    output = torch.tensor(output_numpy, dtype=torch.float32, device=device)

                loss = criterion(output, target.unsqueeze(1))
                test_loss += loss.item()
                predicted = output.round()
                test_correct += (predicted == target.unsqueeze(1)).sum().item()
                test_total += target.size(0)
                test_loader_tqdm.set_postfix({'准确率': 100 * test_correct / test_total, '最优损失epoch': best_epoch,
                                              '最高准确率': best_test_accuracy})

        test_loss /= test_total
        total_test_loss.append(test_loss)
        test_accuracy = 100 * test_correct / test_total
        total_test_accuracy.append(test_accuracy)

        # 早停机制
        if test_accuracy > best_test_accuracy:  # 测试精准率不断上升
            best_test_accuracy = test_accuracy
            best_model = model.state_dict()
            model_save_path = os.path.join(save_path, f'doctor_{i}.pth')
            torch.save(best_model, model_save_path)  # 保存拥有最高准确率的模型
            epochs_without_improvement = 0
        elif best_train_accuracy < train_accuracy < test_accuracy + 5:  # 训练精准率不断上升且未过拟合
            best_train_accuracy = train_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if test_loss < best_test_loss:
            best_epoch = epoch + 1
            best_test_loss = test_loss

        if epochs_without_improvement >= 25:
            print(f"早停触发，在第 {epoch + 1} 轮结束训练.")
            break

        # 实时更新损失图和准确率图
        plot_loss(total_train_loss, total_test_loss, ax_loss, i, n_doctor)
        plot_accuracy(total_train_accuracy, total_test_accuracy, ax_acc, i, n_doctor, best_train_accuracy, best_test_accuracy)

    plt.close(fig)

    # 加载最佳模型
    model.load_state_dict(best_model)

    # 收集最佳模型的预测概率
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in tes_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            predictions.extend(output.cpu().numpy())

    # 将预测结果转换为 NumPy 数组
    predictions = np.concatenate(predictions)

    # 如果 train_with_label 为 True，则将前一层预测结果与本层预测结果加权求和
    if train_with_label:

        result_of_pre_hospital = np.array(result_of_pre_hospital)
        result_of_pre_hospital = result_of_pre_hospital.squeeze(-1)
        predictions = predictions * 0.5 + result_of_pre_hospital * 0.5

    return best_test_accuracy, predictions


def read_data():
    file_path = 'high_diamond_ranked_10min.csv'
    data = pd.read_csv(file_path)

    # 删除 'gameId' 特征
    data.drop('gameId', axis=1, inplace=True)

    # 新增特征：击杀减去死亡
    data['blueKillsMinusDeaths'] = data['blueKills'] - data['blueDeaths']
    data['redKillsMinusDeaths'] = data['redKills'] - data['redDeaths']

    # 新增特征：击杀减去助攻
    data['blueKillsMinusAssists'] = data['blueKills'] - data['blueAssists']
    data['redKillsMinusAssists'] = data['redKills'] - data['redAssists']

    # 删除特征：死亡数
    data.drop(['blueDeaths', 'redDeaths'], axis=1, inplace=True)

    # 将 'blueWins' 放置在最后
    blue_wins = data.pop('blueWins')
    data['blueWins'] = blue_wins

    # 转换为 numpy 数组
    data = data.to_numpy()

    return data


def normalize_data(data):

    # 计算每列的最小值和最大值
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    # 避免除以零，将最大值和最小值相同列的最大值稍微增加一点
    max_vals[max_vals == min_vals] += 0.0001

    # 归一化数据
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def plot_loss(training_loss, testing_loss, ax_loss, i, n_doctor):
    ax_loss.clear()
    ax_loss.plot(training_loss, label='Training Loss')
    ax_loss.plot(testing_loss, label='Testing Loss')
    ax_loss.set_title(f"Loss of doctor {i + 1}/{n_doctor}")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    plt.draw()
    plt.pause(0.01)


def plot_accuracy(train_accuracies, test_accuracies, ax_acc, i, n_doctor, best_train_accuracy, best_test_accuracy):
    ax_acc.clear()
    ax_acc.plot(train_accuracies, label=f'Train Accuracy {best_train_accuracy:.2f}')
    ax_acc.plot(test_accuracies, label=f'Test Accuracy {best_test_accuracy:.2f}')
    ax_acc.set_title(f"Accuracy of doctor {i + 1}/{n_doctor} ")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    plt.draw()
    plt.pause(0.001)


def check_generated_sample_quality(symptoms, diseases, i, n_doctors, nurse):
    generated_sample = np.hstack((symptoms, diseases.reshape(-1, 1)))
    divided_sample, _ = classify_patients(generated_sample, nurse, n_doctors, cluster_with_label=True)
    quality = len(divided_sample[i]) / len(generated_sample)
    print(f"原始样本与生成样本共有{quality}比例隶属于{i + 1}簇")
    return 0


if __name__ == '__main__':
    main()

