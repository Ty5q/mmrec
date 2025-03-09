from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import comb
import json
import os

import numpy as np
import pandas as pd
import random
import scipy.linalg as la
import scipy.spatial.distance as dist
import copy

from sklearn.preprocessing import MaxAbsScaler


def strarray2ndarray(sarray):
    res = []
    for sa in sarray:
        tmp = sa.replace('[', '').replace(']', '').split(',')
        tres = [int(t) for t in tmp]
        res.append(tres)
    return np.array(res)


# 定义读取数据及嵌入向量的函数
def read_embed_data(RAWPATH, dataset):
    # 读取物品点击数据，此数据包含了冷启动物品的索引
    count_data = np.load('{}/{}/{}_item_count.npz'.format(RAWPATH, dataset, dataset))
    cold = count_data['idxes']  # 获取物品索引
    n_items = len(cold)  # 计算物品总数

    # 读取GRU模型的预测数据，包括预测评分和嵌入向量
    gru_data = np.load('../res_{}/{}_GRU4Rec_pred.npz'.format(dataset, dataset))
    # 替换该行以读取BPR模型的预测数据
    # gru_data = np.load('res_{}/{}_bpr_pred.npz'.format(dataset, dataset))
    gru_pred, embed = gru_data['pred'], gru_data['embed']  # 提取预测评分和嵌入向量
    gru_len = len(gru_pred[0])  # 获取预测评分的长度

    # 读取测试集数据
    test_pd = pd.read_csv('{}/{}/test.csv'.format(RAWPATH, dataset), sep='\t')
    ground_truth = test_pd['item_id'].values.reshape((-1, 1))  # 提取每个用户交互的物品ID
    neg_items = strarray2ndarray(test_pd['neg_items'].values)  # 转换负样本数据为NumPy数组
    item_list = np.concatenate((ground_truth, neg_items), axis=1)  # 将真实物品和负样本物品合并
    # 返回嵌入向量、合并后的物品列表和一部分冷启动物品的索引
    return embed, item_list, cold[:int(0.2 * n_items + 1)]


def process_pred(RAWPATH, dataset, method, k):
    if not os.path.exists('../res_{}/k{}'.format(dataset, k)):
        os.mkdir('../res_{}/k{}'.format(dataset, k))
    filename = '../res_{}/k{}/{}_ind_o_{}_{}.csv'.format(dataset, k, dataset, k, method)

    data = np.load('../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method))
    pred = data['pred']

    # 读取ground truth
    # with open('{}/{}/{}_count.json'.format(RAWPATH, dataset, dataset), 'r') as fs:
    #     counts = json.load(fs)
    # test_pd = pd.read_csv('{}/{}/test.csv'.format(RAWPATH, dataset), sep='\t')
    ulist = [str(i) for i in range(1, len(test_df['user_id'].values.tolist()) + 1)]
    # ulist = list(counts.keys())
    for ii, u in enumerate(ulist):
        if u == '3684':
            print('test')

        cur_pred = pred[ii]
        res = []
        res.append(cur_pred[0: k].tolist())

        res_dict = {'user_id': [u], 'ind': [np.array(res)]}
        res_df = pd.DataFrame(res_dict)
        if os.path.exists(filename):
            res_df.to_csv(filename, mode='a', header=False, index=0, sep='\t')
        else:
            res_df.to_csv(filename, index=0, sep='\t')
        # print('{} ok'.format(u))


def str_toarray(strs, t_type):
    r_str = strs[1:-1].split('\n')
    res = []
    for r in r_str:
        tmp = r.replace('array(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        i_tmp = [t_type(t) for t in tmp]
        res.append(i_tmp)
    return np.array(res)


def str_toarray3(strs, t_type, k):
    r_str = strs[1:-1].split('\n')
    res = []
    for r in r_str:
        tmp = r.replace('array(', '').replace(')', '').replace('[', '').replace(']', '').split(',')
        i_tmp = []
        for t in tmp:
            if t == '':
                continue
            i_tmp.append(t_type(t))
            if len(i_tmp) == k:
                res.append(i_tmp)
                i_tmp = []
    return np.array(res)


def str_toarray2(sitems):
    res = []
    for st in sitems:
        tmp = st.replace('[', '').replace(']', '').split(',')
        intt = [int(t) for t in tmp]
        res.append(intt)
    return np.array(res)


def str2ndarray(s):
    res = []
    tmp = s.split('\n')
    for t in tmp:
        lis = t.replace('[', '').replace(']', '').split()
        lt = [int(l) for l in lis]
        res.append(lt)
    return np.array(res)


def str3ndarray(s):
    res = []
    tmp = s.split('\n')
    for t in tmp:
        lis = t.replace('[', '').replace(']', '').split()
        lt = [float(l) for l in lis]
        res.append(lt)
    return np.array(res)


def cal_angle(v1, v2):
    # 计算两个向量之间的夹角
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # 使用向量点积和模长计算夹角的余弦值
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # 确保余弦值在[-1,1]内
    cos_angle = max(min(cos_angle, 1), -1)
    # 计算夹角（以度为单位）
    angle = np.degrees(np.arccos(cos_angle))
    return angle


# 为每个用户选择一个最优解，所选解在目标空间中与向量(1,1,...)的夹角最小，确保所选解在多个目标之间达到平衡
def finalSelection(P, F):
    Pfinal = []
    objective_space_vector = np.ones(len(F[0]))
    objective_space_vector[0] = -1
    # objective_space_vector[0] = -1/10
    objective_space_vector[1] = -1
    # objective_space_vector[1] = -1/10
    # objective_space_vector[2] = 1/10
    angle = []
    for idx, p in enumerate(P):
        cur_angle = cal_angle(F[idx], objective_space_vector)
        angle.append(cur_angle)
    top = np.argsort(angle)
    for i in top:
        # for t in P[i]:
        #     if t not in Pfinal:
        #         Pfinal.append(t)
        #     if len(Pfinal) == 20:
        #         break
        Pfinal.append(P[i])
    return Pfinal


# 距离
def finalSelection1(P, F):
    Pfinal = []
    objective_space_vector = np.ones(len(F[0]))
    objective_space_vector[0] = -1
    objective_space_vector[1] = -1
    # objective_space_vector[2] = 0
    dis = []
    for idx, p in enumerate(P):
        cur_dis = np.linalg.norm(F[idx] - objective_space_vector)
        dis.append(cur_dis)
    top = np.argsort(dis)
    for i in top:
        # for t in P[i]:
        #     if t not in Pfinal:
        #         Pfinal.append(t)
        #     if len(Pfinal) == 20:
        #         break
        Pfinal.append(P[i])
    return Pfinal


from kneed import KneeLocator
# knee
def finalSelection2(P, F):
    Pfinal = []
    objective_space_vector = np.ones(len(F[0]))
    objective_space_vector[0] = -1
    objective_space_vector[1] = -1
    # objective_space_vector[2] = 0
    dis = []
    for idx, p in enumerate(P):
        cur_dis = np.linalg.norm(F[idx] - objective_space_vector)
        dis.append(cur_dis)

    # 使用KneeLocator找到knee point
    knee_locator = KneeLocator(range(len(dis)), dis, curve='convex', direction='increasing')
    knee_index = knee_locator.knee  # 这是knee point的索引

    for t in P[: knee_index + 1]:
        Pfinal.append(t)
    return Pfinal


# 权衡解
def bal_rec(RAWPATH, dataset, k, method, cnt):
    if not os.path.exists('../res_{}/k{}'.format(dataset, k)):
        os.mkdir('../res_{}/k{}'.format(dataset, k))
    # 存结果的文件
    filename = '../res_{}/k{}/{}_{}_res_bal_{}_test_2_1.csv'.format(dataset, k, dataset, method, k)
    # 读取种群数据
    moea_pop = '../res_{}/{}_{}_moea_{}_2.csv'.format(dataset, dataset, method, k)
    # moea_pop = '../res_{}/{}_{}_moea_{}_4.csv'.format(dataset, dataset, method, k)
    # moea_pop = '../res_{}/{}_{}_moea_{}_6.csv'.format(dataset, dataset, method, k)

    assert os.path.exists(moea_pop), '缺失种群结果'
    res = pd.read_csv(moea_pop, sep='\t')

    # 读取ground truth
    # with open('{}/{}/{}_count.json'.format(RAWPATH, dataset, dataset), 'r') as fs:
    #     counts = json.load(fs)
    # with open('{}/{}/{}_ground_truth.json'.format(RAWPATH, dataset, dataset), 'r') as fs:
    #     gts = json.load(fs)

    # 读取gru 预测信息
    gru_data = np.load('../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method))
    gru_pred, embed = gru_data['pred'], gru_data['embed']
    gru_len = len(gru_pred[0])

    # 原始的test数据
    test_pd = pd.read_csv('{}/{}/test.csv'.format(RAWPATH, dataset), sep='\t')
    ground_truth = test_pd['item_id'].values.reshape((-1, 1))
    neg_items = str_toarray2(test_pd['neg_items'].values)
    item_list = np.concatenate((ground_truth, neg_items), axis=1)
    embed_list = np.array([embed[x] for x in item_list])

    user_list = test_df['user_id'].values
    for i_idx, idx in enumerate(user_list):
        # 获取当前用户的item列表和结果
        cur_item_list = item_list[i_idx]

        # 单任务展平
        res['user_id'] = res['user_id'].apply(lambda x: int(x.strip('[]')) if isinstance(x, str) else x)

        cur_res = res.loc[res['user_id'] == (i_idx + 1)]
        pop, fit = cur_res['pop'].values[0], cur_res['fit'].values[0]
        p_pop = str_toarray3(pop, int, k)
        f_fit = str_toarray3(fit, float, 3)
        # f_fit = str3ndarray(fit)

        # 进行finalSelection
        # print(len(f_fit))
        final_pop = finalSelection(p_pop, f_fit)[0]

        res_dict = {'user_id': [idx], 'ind': [np.array(final_pop)]}
        res_df = pd.DataFrame(res_dict)
        if os.path.exists(filename):
            res_df.to_csv(filename, mode='a', header=False, index=0, sep='\t')
        else:
            res_df.to_csv(filename, index=0, sep='\t')
        # print('{} user {} saved'.format(i_idx, idx))


def cal_div(real_rec_ind):
    tmp_dict = {}
    for i in range(cluster_sum):
        tmp_dict[i] = 0
    for i in real_rec_ind:
        idx = item_meta_df.iloc[i - 1, 1:].to_numpy().nonzero()[0]
        for t in idx:
            tmp_dict[t] = 1
    cnt = sum(1 for value in tmp_dict.values() if value == 1)
    return cnt / cluster_sum

import json

# new
def process_epsilon_pred(RAWPATH, dataset, method1, method2, k):
    filename = '../res_{}/k{}/{}_epsilon.csv'.format(dataset, k, dataset)

    data = np.load('../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method1))
    other_pred = np.load('../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method2))['pred']
    pred = data['pred']
    with open('{}/{}/{}_count.json'.format(RAWPATH, dataset, dataset)) as fs:
        counts = json.load(fs)
    ulist = list(counts.keys())
    for ii, u in enumerate(ulist):
        if u == '3684':
            print('test')

        cur_pred = pred[ii]

        start = 0
        other = other_pred[ii]
        re_list = []
        for e in range(k):
            epsilon = random.random()
            if epsilon < 0.7:
                re_list.append(cur_pred[start])
                start += 1
            else:
                re_list.append(other[random.randint(0, len(other)-1)])

        res_dict = {'user_id': [u], 'ind': [np.array(re_list)]}
        res_df = pd.DataFrame(res_dict)
        if os.path.exists(filename):
            res_df.to_csv(filename, mode='a', header=False, index=0, sep='\t')
        else:
            res_df.to_csv(filename, index=0, sep='\t')
        # print('{} ok'.format(u))


def MMR(itemList, itemScoreList, similarityMatrix, topN, lambdaConstant = 0.7):
    s, r = [], itemList.tolist()
    while len(s) < topN:
        score = 0
        selectOne = None
        for i in r:
            firstPart = itemScoreList[0, i]
            secondPart = 0
            for j in s:
                ma = np.linalg.norm(similarityMatrix[i, :])
                mb = np.linalg.norm(similarityMatrix[j, :])
                sim2 = (np.matmul(similarityMatrix[i, :], similarityMatrix[j, :]))/(ma*mb)
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = lambdaConstant * firstPart - (1 - lambdaConstant) * secondPart
            if equationScore > score:
                score = equationScore
                selectOne = i
        if selectOne == None:
            selectOne = i
        r.remove(selectOne)
        s.append(selectOne)
    return (s, s[:topN])[topN > len(s)]


def process_mmr_pred(RAWPATH, dataset, method, k):
    filename = '../res_{}/k{}/{}_mmr.csv'.format(dataset, k, dataset)

    data = np.load('../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method))
    embed, item_list, cold_list = read_embed_data(RAWPATH, dataset)
    pred = data['pred']
    score = data['score']

    with open('{}/{}/{}_count.json'.format(RAWPATH, dataset, dataset)) as fs:
        counts = json.load(fs)
    ulist = list(counts.keys())
    for ii, u in enumerate(ulist):
        if u == '3684':
            print('test')
        cur_pred = pred[ii]
        cur_score = score[ii]
        re_curscore = cur_score.reshape(-1, 1)
        min_max_scaler = MaxAbsScaler().fit(re_curscore)
        score_minmax = min_max_scaler.transform(re_curscore)
        scorelist = score_minmax.T
        i_embed = np.array([embed[item_list[ii, j]] for j in range(len(item_list[0]))])

        res_all = MMR(cur_pred, scorelist, i_embed, k*20)
        res = [res_all[i: i + k] for i in range(0, len(res_all), k)]

        res_dict = {'user_id': [u], 'ind': [np.array(res)]}
        res_df = pd.DataFrame(res_dict)
        if os.path.exists(filename):
            res_df.to_csv(filename, mode='a', header=False, index=0, sep='\t')
        else:
            res_df.to_csv(filename, index=0, sep='\t')
        # print('{} ok'.format(u))


# 定义评估函数
def evaluate(RAWPATH, dataset, k, method, cnt):
    # 原始的test数据
    test_pd = pd.read_csv('{}/{}/test.csv'.format(RAWPATH, dataset), sep='\t')
    ground_truth = test_pd['item_id'].values.reshape((-1, 1))

    neg_items = str_toarray2(test_pd['neg_items'].values)
    item_list = np.concatenate((ground_truth, neg_items), axis=1)

    # 读取多目标推荐结果数据
    # result_df = pd.read_csv('../res_{}/k{}/{}_pop.csv'.format(dataset, k, dataset), sep='\t')
    # result_df = pd.read_csv('../res_{}/k{}/{}_{}_res.csv'.format(dataset, k, dataset, method), sep='\t')

    # result_df = pd.read_csv('../res_{}/k{}/{}_ind_o_{}_{}.csv'.format(dataset, k, dataset, k, method), sep='\t')
    result_df = pd.read_csv('../res_{}/k{}/emmr_{}_{}_{}.csv'.format(dataset, k, dataset, method, k), sep='\t')
    # result_df = pd.read_csv('../res_{}/k{}/{}_{}_res_bal_{}_test_30_vec_2_10_dis.csv'.format(dataset, k, dataset, method, k), sep='\t')

    # result_df = pd.read_csv('../res_{}/k{}/{}_epsilon.csv'.format(dataset, k, dataset), sep='\t')
    # result_df = pd.read_csv('../res_{}/k{}/{}_mmr.csv'.format(dataset, k, dataset), sep='\t')

    # 用户列表
    user_list = result_df['user_id'].values
    # 用户数量
    n_users = len(user_list)

    # 从结果中读取推荐的索引
    rec_inds = result_df['ind'].values

    # 初始化用于存放各种评估指标的数组
    # recall = np.zeros((n_users, 20), dtype=np.float32)
    # ndcg = np.zeros((n_users, 20), dtype=np.float32)
    # hr = np.zeros((n_users, 20), dtype=np.float32)
    # div = np.zeros((n_users, 20), dtype=np.float32)
    # nov = np.zeros((n_users, 20), dtype=np.float32)
    recall = []
    ndcg = []
    hr = []
    div = []

    # 遍历每一个用户
    for i, idx in enumerate(user_list):
        # ground_truth = list(gts.values())[i]
        # 获取用户推荐列表中每个物品的嵌入向量
        # i_embed = np.array([embed[item_list[i, j]] for j in range(len(item_list[0]))])

        # 打印当前处理用户ID
        # print(idx)
        # 将推荐列表从字符串转换成NumPy数组
        i_rec_ind = str2ndarray(rec_inds[i])
        # i_rec_ind = str_toarray3(rec_inds[i], int, k)
        # if k == 20:
        #     tmp = []
        #     for t in i_rec_ind:
        #         tmp += t
        #     i_rec_ind = np.array([tmp])
        real_rec_ind = item_list[i][i_rec_ind[0].flatten()]  # 转换为真实的物体
        # real_rec_ind = i_rec_ind[0]  # 转换为真实的物体

        # 评估
        # hr.append(int(any(item in real_rec_ind for item in truth_item)))
        # recall.append(len(set(real_rec_ind) & set(truth_item)) / float(len(truth_item)))
        # dcg = 0.0
        # for j, it in enumerate(real_rec_ind):
        #     if it in truth_item:
        #         dcg += 1 / np.log2(j + 2)
        # idcg = sum(1 / np.log2(z + 2) for z in range(min(len(truth_item), k)))
        # ndcg.append(dcg / idcg)

        gt_num = len(i_rec_ind)
        hs, ns = [], []
        for r in i_rec_ind:
            tmp = np.where(r < gt_num)[0]
            if len(tmp) == 0:
                hs.append(0)
                ns.append(0)
            else:
                gt_rank = tmp[0] + 1
                hs.append(1)
                ns.append(1 / np.log2(gt_rank + 1))
        hr.append(np.mean(hs))
        ndcg.append(np.mean(ns))

        r_loc = i_rec_ind[i_rec_ind < gt_num]
        recall.append((len(np.unique(r_loc)) / gt_num))

        # diversity
        tmp_div = cal_div(real_rec_ind)
        div.append(tmp_div)

    # 打印各指标的平均值
    print('hr: {}'.format(np.mean(hr)))
    # print('recall: {}'.format(np.mean(recall)))
    print('ndcg: {}'.format(np.mean(ndcg)))
    print('diversity: {}'.format(np.mean(div)))

    # 最后打印测试结束信息
    print("test")


if __name__ == '__main__':
    # 读入数据集
    dataset = 'Grocery_and_Gourmet_Food'   # 'ml-1m' 'Grocery_and_Gourmet_Food'
    method = 'GRU4Rec'  # 'GRU4Rec' 'ComiRec' 'SLRCPlus' 'SASRec' 'ContraRec' 'TiSASRec' 'KDA' 'Caser' 'NARM'
    path = r'../../data/' + dataset
    test_df = pd.read_csv(path + r'/test.csv', sep='\t')
    item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')
    train_df = pd.read_csv(path + r'/train.csv', sep='\t')

    RAWPATH = '../../data'

    cluster_sum = item_meta_df.shape[1] - 1
    var = [5,10]

    cnt = 5

    for k in var:
        # process_pred(path, dataset, method, k)
        # process_mmr_pred(RAWPATH, dataset, method, k)
        # process_epsilon_pred(RAWPATH, dataset, method, 'SASRec', k)
        bal_rec(RAWPATH, dataset, k, method, cnt)
        print('------------------------------------------------')
        print(f'--------------------@{k}-----------------------')
        evaluate(RAWPATH, dataset, k, method, cnt)
        print(f'--------------------@{k}-----------------------')
        print('------------------------------------------------')

