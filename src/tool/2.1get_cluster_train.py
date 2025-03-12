import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# 读入数据集
dataset = 'ten-1m'  # 'ml-1m' 'Montana' 'ten-1m'
path = r'../../data/' + dataset
train_df = pd.read_csv(path + r'/train.csv', sep='\t')

if dataset != 'ten-1m':
    item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')

    # print(item_meta_df.columns.tolist())

    cluster_sum = len(item_meta_df.iloc[0, 1:])


    def cal_user_class(item_list_i, col_name):
        class_list = []
        temp = {}
        cnt = []
        for i in range(len(col_name)):
            cnt.append(0)

        for i in item_list_i:
            # print(i)
            target_row = item_meta_df[item_meta_df['item_id'] == i]
            cols_non_zero = target_row.iloc[0][target_row.iloc[0] != 0].index.tolist()
            cols_non_zero_indices = [target_row.columns.get_loc(col) for col in cols_non_zero]
            for j in cols_non_zero_indices[1: -1]:
                if j not in class_list:
                    class_list.append(j)
                cnt[j - 1] += 1
                # cnt[j - 1] = 1
        for idx, n in enumerate(col_name):
            temp[n] = []
            temp[n].append(cnt[idx])
        class_list.sort()
        return class_list, temp

    out_df = train_df.copy()
    user_list = set(out_df['user_id'].values)

    col_name = item_meta_df.columns.tolist()[1:-1]

    data = {}
    data['user_id'] = []
    data['item_id'] = []

    data['diversity'] = []
    # data['class'] = []
    for k in col_name:
        data[k] = []
    for user in range(1, len(user_list) + 1):
        item_list_i = list(set(out_df.loc[out_df['user_id'] == user]['item_id'].values))

        data['user_id'].append(user)
        data['item_id'].append(item_list_i)

        class_list, temp = cal_user_class(item_list_i, col_name)
        diversity = len(class_list) / cluster_sum
        for k in col_name:
            data[k].append(temp[k][0])
        # 得到每个用户访问项目所占类别数，然后除以总类别数，得到该用户的多样性
        data['diversity'].append(diversity)
        # data['class'].append(class_list)
        print(f"user {user} 处理完成")


    datadf = pd.DataFrame(data)
    datadf.to_csv(path + r'/train1.csv', index=False, sep='\t')
else:
    # 初始化数据字典
    data = {
        'user_id': [],
        'item_id': []
    }

    # 获取所有用户ID
    user_list = set(train_df['user_id'].values)

    # 遍历每个用户，收集其交互的项目列表
    for user in user_list:
        item_list_i = list(set(train_df[train_df['user_id'] == user]['item_id'].values))
        data['user_id'].append(user)
        data['item_id'].append(item_list_i)
        print(f"user {user} 处理完成")

    # 将数据转换为DataFrame并保存为新的CSV文件
    datadf = pd.DataFrame(data)
    datadf.to_csv(path + r'/train1.csv', index=False, sep='\t')
