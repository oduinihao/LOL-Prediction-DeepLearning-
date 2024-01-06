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
    data_array = data.to_numpy()

    print(data_array.shape)

    return data_array

read_and_process_data()

