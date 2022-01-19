import random

import numpy as np

fold = 5
m = 708
n = 1512
A = np.loadtxt("dataset/mat_drug_protein.txt")   # A矩阵  行是药物 列是蛋白质

A_dim1 = A.flatten()                # 708*1512
i = 0
list_1 = []
while i < len(A_dim1):
    if A_dim1[i] == 1:
        list_1.append(i)           # list_1列表存放所有1在labels数组中的位置
    i = i+1
num1 = len(list_1)                   # 得到1的个数
group_size1 = int(num1/fold)          # 计算得到每组数据中1 的个数  : 2
random.seed(10)
random.shuffle(list_1)             # 将1的位置随机打乱

# 将1分为5组， 存放在grouped_data1[5*group_size]中
array_1 = np.array(list_1)[0:fold*group_size1]            # 舍弃最后的几个1
grouped_data1 = np.reshape(array_1, (fold, group_size1))
np.savetxt("result/DTI_index_1.txt", grouped_data1, fmt='%d')      # 5*384

i = 0
list_0 = []
while i < len(A_dim1):
    if A_dim1[i] == 0:
        list_0.append(i)           # list_0列表存放所有1在labels数组中的位置
    i = i+1
num0 = len(list_0)                   # 得到0的个数
group_size0 = int(num0/fold)          # 计算得到每组数据中0 的个数  : 2
random.seed(10)
random.shuffle(list_0)             # 将0的位置随机打乱

# 将0分为5组， 存放在grouped_data0[5*group_size]中
array_0 = np.array(list_0)[0:fold*group_size0]            # 舍弃最后的几个1
grouped_data0 = np.reshape(array_0, (fold, group_size0))
np.savetxt("result/DTI_index_0.txt", grouped_data0, fmt='%d')      # 5*384

# i = 0
# list_5 = []
# while i < len(A_dim1):
#     if A_dim1[i] == 0.5:
#         list_5.append(i)           # list_0列表存放所有1在labels数组中的位置
#     i = i+1
# num5 = len(list_5)                   # 得到0的个数
# random.seed(10)
# random.shuffle(list_5)             # 将0的位置随机打乱
# np.savetxt("result/index_5.txt", list_5, fmt='%d')      # 5*384