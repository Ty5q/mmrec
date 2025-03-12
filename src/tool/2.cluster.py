import time

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


from kneed import KneeLocator
# 找到最佳聚类中心点
def find_best(x, df):
    # 聚类
    best_k = -1
    best_score = -1
    silhouette_scores = []
    cluster_centers = []

    km = KMeans(n_clusters=2, random_state=42)

    labels = km.fit_predict(x)

    silhouette_avg = silhouette_score(x, labels)
    silhouette_scores.append(silhouette_avg)

    if silhouette_avg > best_score:
        df['cluster_label'] = km.labels_
        cluster_centers = km.cluster_centers_
        best_score = silhouette_avg
        best_k = 6

    # print(f"最佳聚类数目k为：{best_k}")
    return best_k, cluster_centers, silhouette_scores, df


# 找到离聚类中心最近的点代替该类别
def find_closest(df, cluster_centers):
    cluster = []
    # 找到离聚类中心最近的点用以代替该类别
    closest_points_indices = []
    for i, c in enumerate(cluster_centers):
        max_d = -1
        index_list = df.index[df['cluster_label'] == i].tolist()
        for idx in index_list:
            df.loc[idx, 'distance'] = np.abs(df.iloc[idx, 2: 2 + len(c[1:])].values - c[1:]).sum() / len(c[1:])
            if df.loc[idx, 'distance'] > max_d:
                max_d = df.loc[idx, 'distance']
        df.loc[index_list, 'penc'] = df.loc[index_list, 'distance'] / max_d
        closest_points_indices.append(np.argmin(df.loc[index_list, 'distance']))
        cluster.append(i)
    return closest_points_indices, cluster, df


# 找到最佳聚类cluster数以及聚类中心
def find_best_cluster(df):
    # features = df.columns.tolist()[1:-1]
    # no_div
    id = df.columns.get_loc("emb_0")
    features = df.columns.tolist()[id:-1]
    x = df[features]
    # KMeans聚类
    best, cluster_centers, silhouette_scores, df = find_best(x, df)

    # 计算每个用户到聚类中心的距离
    closest_points_indices, cluster, df = find_closest(df, cluster_centers)

    # 保存每个用户的标签
    df.to_csv(path + "/cluster_label_train1_info_vec_2.csv", sep='\t')

    # 获取这些最近点的信息
    closest_user_ids = df.loc[closest_points_indices, 'user_id']
    closest_item_ids = df.loc[closest_points_indices, 'item_id']

    # 保存聚类中心的信息
    cluster_center_info = pd.DataFrame({
        'user_id': closest_user_ids,
        'item_id': closest_item_ids,
        'cluster': cluster,
    })


    cluster_center_info.to_csv(path + "/cluster_center_info_vec_2.csv", sep='\t')
    print('聚类中心信息已保存')


if __name__ == '__main__':
    dataset = 'ten-1m'   # 'ml-1m' 'Montana' 'ten-1m'
    path = r'../../data/' + dataset

    train_df = pd.read_csv(path + '/pre_cluster.csv', sep='\t')

    find_best_cluster(train_df)
