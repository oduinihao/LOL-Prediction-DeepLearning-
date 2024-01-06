# LOL-Prediction-DeepLearning-
challenge on high_diamond_ranked_10min data
此项目为英雄联盟对局结果预测实验代码，在不使用决策树类模型的前提下进行，使用多种方法，最优模型为MLP，配合SMOTE数据增强下预测准确率最高为**75.2%**，代码在MLP_SMOTE.py文件中。因为训练具有一定随机性，可能需要进行若干次训练。
此外，本实验还尝试了其他多种方法：
1.SVM支持向量机：SVM.py
2.MLP网络：MLP.py
3.MLP网络与SMOTE数组增强：MLP_SMOTE.py
4.TabNET网络与SMOTE数据增强：TabNet_SMOTE.py
5.引入MultiHeadAttention的MLP网络与SMOTE数据增强：MLP_MSA_SMOTE.py
6.MLP与知识蒸馏相结合（教师网络为SVM支持向量机，学生网络为MLP多层感知机）预测精确率可以稳定在74%：Distillation\Student_MLP.py 
7.一种基于K-means数据分类的MLP算法，K-means进行聚类后观察每一簇的蓝色方胜率，若该簇的蓝色方胜率大于阈值a（75%）且测试集划分结果与训练集聚类结果近似时，直接将归为此簇的所有样本认定为蓝色方胜利，反之亦然。当蓝色方大于1-a且小于a时，针对该簇数据训练MLP网络进行分类。整体结构与医院分诊台类似，代码可直接一键运行：Hospital\Hospital.py
