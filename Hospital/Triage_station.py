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
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
'''
cluster_ap(data)
cluster_birch(k, data)
cluster_dbscan(data)
cluster_m_kmeans(k, data)
cluster_optics(data)
cluster_spectral(k, data)
cluster_g_mixture(k, data)
cluster_kmeans(k, data)
cluster_meanshift(data)
from sklearn.cluster import MeanShift
'''


def main():
    lol_data = read_and_process_data()
    lol_data = normalize_data(lol_data)
    train_nurse_kmeans(5, lol_data)


def classify_patients(patients, model, result_of_pre_hospital=None, cluster_with_label=False):
    # 分离特征和原始标签
    symptoms = patients[:, :-1]

    # 填充标签(决定使用标签进行聚类时)
    if cluster_with_label:
        if result_of_pre_hospital is None:
            pre_diagnosis = np.full(len(patients), 0.5)
        else:
            pre_diagnosis = result_of_pre_hospital

        symptoms_with_diagnosis = np.column_stack((symptoms, pre_diagnosis))  # 合并特征和临时标签
        doctors = model.predict(symptoms_with_diagnosis)
    else:
        doctors = model.predict(symptoms)

    if result_of_pre_hospital is not None:  # 返回42个特征，同时分割上层的结果
        patients = np.column_stack((patients, result_of_pre_hospital))

    divided_patients = {}
    total_prevalence_rate = []

    for doctor_togo in np.unique(doctors):
        division = patients[doctors == doctor_togo]  # 选择当前簇的所有数据
        prevalence_rate = np.mean(division[:, -2])  # 计算真实标签的平均值
        total_prevalence_rate.append(prevalence_rate)
        divided_patients[f"class_{doctor_togo}"] = division

        patients_shape = division.shape
        print(f"簇 {doctor_togo + 1} 形状: {patients_shape}, 蓝方胜率: {prevalence_rate:.2f}")

    return list(divided_patients.values()), total_prevalence_rate


def train_nurse_kmeans(n_doctors, patient, cluster_with_label=False):
    print("开始kmeans聚类！")
    # 分离特征和原始标签
    symptoms = patient[:, :-1]
    diseases = patient[:, -1]

    # 携带标签进行分类
    if cluster_with_label:
        symptoms = patient
        nurse = KMeans(n_clusters=n_doctors, random_state=0).fit(symptoms)
    else:
        nurse = KMeans(n_clusters=n_doctors, random_state=0).fit(symptoms)

    doctors = nurse.labels_
    divided_patients = {}
    total_accuracy = []

    for doctor_togo in np.unique(doctors):
        division = patient[doctors == doctor_togo]  # 选择当前簇的所有数据
        prevalence_rate = np.mean(division[:, -1])  # 计算真实标签的平均值
        total_accuracy.append(prevalence_rate)
        divided_patients[f"cluster_{doctor_togo}"] = division

        patients_shape = division.shape
        print(f"簇 {doctor_togo + 1} 形状: {patients_shape}, 蓝方胜率: {prevalence_rate:.2f}")

    return [divided_patients[f"cluster_{i}"] for i in np.unique(doctors)], nurse, total_accuracy


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

    # 分离特征和标签
    data = data[:int(len(data) * 0.8)]

    return data


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


if __name__ == '__main__':
    main()