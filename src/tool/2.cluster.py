import time

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# 计算项目流行度
def cal_p(df):
    out = {}
    for i in range(0, len(df)):
        temp = df['item_id'][i]
        if temp not in out.keys():
            out[temp] = 1
        else:
            out[temp] += 1
    return out


# 定义多样性
def calc_diversity(idx, df, cluster_sum):
    return item_meta_df.iloc[df.iloc[idx, 1] - 1, 1:].sum() / cluster_sum


# 定义新颖度
def calc_novelty(idx, df, p):
    temp = item_meta_df.loc[df.iloc[idx, 1] - 1, 'item_id']
    if temp not in p.keys():
        return 1
    else:
        return 1 / p[temp]


# 计算项目的多样性和新颖性
def cal_sum(df, cluster_sum):
    # 计算项目的多样性
    for i in range(0, len(df)):
        df.loc[i, 'diversity'] = calc_diversity(i, df, cluster_sum)

    # 计算项目流行度
    p_dict = cal_p(df)

    # 计算项目的新颖性
    for i in range(0, len(df)):
        df.loc[i, 'novelty'] = calc_novelty(i, df, p_dict)

    return df


# 计算两点间的聚类
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# 找到最高平均silhouette和聚类中心点
def find_best_sil(x, df, task, k_min=2, k_max=30):
    # 聚类
    best_k = -1
    best_score = -1
    silhouette_scores = []
    cluster_centers = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42)

        labels = km.fit_predict(x)

        silhouette_avg = silhouette_score(x, labels)
        silhouette_scores.append(silhouette_avg)

        # time.sleep(0.001)

        if silhouette_avg > best_score:
            df[task + '_cluster_label'] = km.labels_
            cluster_centers = km.cluster_centers_
            best_score = silhouette_avg
            best_k = k
    print(f"{task}最佳聚类数目k为：{best_k}")
    return best_k, cluster_centers, silhouette_scores, df


from sklearn.model_selection import cross_val_score

# 找到离聚类中心最近的点代替该类别
def find_closest(df, cluster_centers, task):
    cluster = []
    # 找到离聚类中心最近的点用以代替该类别
    closest_points_indices = []
    remaining_df = df.copy()  # 创建一个副本以免更改原始数据

    for i, c in enumerate(cluster_centers):
        max_d = -1
        index_list = df.index[df[task + '_cluster_label'] == i].tolist()
        for idx in index_list:
            df.loc[idx, 'distance'] = np.abs(df.iloc[idx, 3: 3 + len(c)].values - c).sum() / len(c)
            if df.loc[idx, 'distance'] > max_d:
                max_d = df.loc[idx, 'distance']
        df.loc[index_list, 'penc'] = df.loc[index_list, 'distance'] / max_d
        closest_points_indices.append(np.argmin(df.loc[index_list, 'distance']))
        # closest_index = np.argmin(np.abs(remaining_df[task] - c))
        # label = int(remaining_df.iloc[closest_index][task + '_cluster_label'])
        # while label in cluster:
        #     # 删除已经选中的用户
        #     remaining_df.drop(remaining_df.index[closest_index], inplace=True)
        #     # 重新计算
        #     closest_index = np.argmin(np.abs(remaining_df[task] - c))
        #     label = int(remaining_df.iloc[closest_index][task + '_cluster_label'])
        #
        # closest_points_indices.append(remaining_df.index[closest_index])
        #
        # cluster.append(label)
        cluster.append(i)
    return closest_points_indices, cluster, df


# 找到最佳聚类cluster数以及聚类中心
def find_best_cluster(df, k_min=2, k_max=30):
    features = df.columns.tolist()[1:]
    x = df[features]
    # # KMeans聚类
    best_d, cluster_centers_d, silhouette_scores_d, df = find_best_sil(x, df, 'diversity')
    # # 找到novelty的最佳聚类
    # best_n, cluster_centers_n, silhouette_scores_n, df = find_best_sil(x_n, df, 'novelty')
    # 计算轮廓系数
    sil_score = silhouette_score(x, cluster_centers_d)
    print("Silhouette Score:", sil_score)
    # 可视化silhouette
    plt.figure()
    plt.plot(range(k_min, k_max + 1), silhouette_scores_d, marker='o', label='diversity')
    # plt.plot(range(k_min, k_max + 1), silhouette_scores_n, marker='.', label='novelty')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Coefficient')
    plt.show()

    # 找到离聚类中心最近的点代替该cluster
    closest_points_indices_d, cluster_d, df = find_closest(df, cluster_centers_d, 'diversity')
    # closest_points_indices_n, cluster_n, _ = find_closest(remaining_df, cluster_centers_n, 'novelty')
    # closest_points_indices = closest_points_indices_d + closest_points_indices_n
    closest_points_indices = closest_points_indices_d
    df.to_csv(path + "/cluster_label_train1_info.csv", sep='\t')
    # cluster = cluster_d + cluster_n
    cluster = cluster_d
    # if_diversity = [0 for _ in range(len(cluster))]
    # if_diversity[:len(cluster_d)] = [1] * len(cluster_d)

    # 获取这些最近点的信息
    closest_user_ids = df.loc[closest_points_indices, 'user_id']
    closest_item_ids = df.loc[closest_points_indices, 'item_id']
    # closest_interaction_times = df.loc[closest_points_indices, 'time']
    closest_diversity = df.loc[closest_points_indices, 'diversity']
    # closest_novelty = df.loc[closest_points_indices, 'novelty']

    cluster_center_info = pd.DataFrame({
        'user_id': closest_user_ids,
        'item_id': closest_item_ids,
        # 'time': closest_interaction_times,
        # 'diversity': closest_diversity,
        # 'novelty': closest_novelty,
        'cluster': cluster,
        # 'if_diversity': if_diversity
    })

    cluster_center_info.to_csv(path + "/cluster_center_info.csv", sep='\t')
    print('聚类中心信息已保存')


def run(df):
    find_best_cluster(df)


if __name__ == '__main__':
    # dataset = 'ml-1m'
    dataset = 'Digital_Music'   # 'Digital_Music' 'Office_Products' 'Clothing_Shoes_and_Jewelry' 'Kindle_Store' 'Pet_Supplies' 'Beauty'
    path = r'../../data/' + dataset

    item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')
    train_df = pd.read_csv(path + r'/train1.csv', sep='\t')

    cluster_sum = item_meta_df.shape[1] - 1
    run(train_df)
