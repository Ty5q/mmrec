import ast
import time
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.interface import sample
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetric
import numpy as np
from pymoo.core.repair import Repair


def str_toarray(sitems):
    res = []
    for st in sitems:
        tmp = st.replace('[', '').replace(']', '').split(',')
        intt = [int(t) for t in tmp]
        res.append(intt)
    return np.array(res)


def select_ind(pop):
    res = []
    for i, ind in enumerate(pop):
        if 0 in ind:
            res.append(i)
    if len(res) >= 1:
        s_id = random.sample(res, 1)
    else:
        s_id = -1
    return s_id


class Item:  # 每个物体的类
    def __init__(self, idx, true_num, acc_score, embed):
        self.idx = idx  # 物品在该用户物品里的编号
        self.true_num = true_num  # 物品在整个数据集里的真实编号
        self.acc_score = acc_score   # [第一个目标] gru 给该物体的序号，0最大，99最小，越小越好
        self.embed = embed


# 对应的repair
class MyRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        if type(pop).__name__ == "Population":
            n = len(pop[0].X)
            for k in range(len(pop)):
                xs = pop[k].X

                ctmp = []
                for j in range(n):
                    # 保证个体内物体不能重复
                    while xs[j] in ctmp:
                        xs[j] = np.random.randint(problem.xl, problem.xu, 1)[0]
                    ctmp.append(xs[j])
            return pop
        else:
            n = len(pop.X)
            xs = pop.X
            ctmp = []
            for j in range(n):
                # 保证个体内物体不能重复
                while xs[j] in ctmp:
                    xs[j] = np.random.randint(problem.xl, problem.xu, 1)[0]
                ctmp.append(xs[j])
            return pop


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


def cal_div_Shannon(real_rec_ind):
    # 丰富度
    tmp_dict = {}
    for i in range(cluster_sum):
        tmp_dict[i] = 0
    for i in real_rec_ind:
        idx = item_meta_df.iloc[i - 1, 1:].to_numpy().nonzero()[0]
        for t in idx:
            tmp_dict[t] += 1
    cnt = sum(1 for value in tmp_dict.values() if value != 0)
    # 均匀度


# 多目标优化
# 定义优化问题
class MyProblem(Problem):

    def __init__(self, user_id, items, xl, xu, k=6):
        super().__init__(n_var=k, n_obj=2, n_constr=0, xl=xl, xu=xu)
        self.items = items
        self.user_id = user_id

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for xi in x:
            acc = 0
            real_rec_ind = item_list[self.user_id - 1][xi]
            # 计算diversity
            diversity = cal_div(real_rec_ind)
            for item_id in xi:
                # acc
                acc += self.items[item_id].acc_score
            acc = acc / len(xi)
            F.append([-diversity, acc])
        out["F"] = np.array(F)
        out["G"] = None



def solve(uid, items, if_center, k=5, xl=0, xu=99, gen=30, popsize=100):
    sampling = get_sampling("int_random")
    init_X = sample(sampling, popsize, k, xl=xl, xu=xu)


    problem = MyProblem(items=items, user_id=uid, k=k, xl=xl, xu=xu)
    algorithm = AGEMOEA(pop_size=popsize,
                        sampling=init_X,
                        crossover=get_crossover("int_sbx", eta=20, prob=1),
                        mutation=get_mutation("int_pm", prob=1, eta=20),
                        eliminate_duplicates=False,
                        repair=MyRepair()
                        )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', gen),
                   save_history=True,
                   verbose=False)

    pop = []
    for r in res.algorithm.pop.get("X"):
        tmp = []
        for rr in r:
            t = int(rr)
            tmp.append(items[t].idx)
            # tmp.append(t)
        pop.append(tmp)

    fit = res.algorithm.pop.get("F")
    seq = np.argsort(fit, axis=0)
    return pop, fit, seq

# 计算项目流行度
def cal_p(df):
    out = {}
    for i in range(1, len(item_meta_df['item_id'])):
        temp = df[df['item_id'] == i]
        if i not in out.keys():
            out[i] = len(temp['item_id'])
    return out


if __name__ == '__main__':
    dataset = 'ml-1m'   # 'ml-1m' 'Montana' 'ten-1m'
    method = 'GRU4Rec'  # 'GRU4Rec' 'ComiRec' 'SLRCPlus' 'SASRec'
    # 读入数据集
    path = r'../../data/' + dataset
    # path = r'../../data/RecSys'
    dev_df = pd.read_csv(path + r'/dev.csv', sep='\t')
    test_df = pd.read_csv(path + r'/test.csv', sep='\t')
    item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')
    # item_meta_df = pd.read_csv(path + r'/item_meta_1.csv', sep='\t')
    train_df = pd.read_csv(path + r'/train1.csv', sep='\t')

    # 计算项目流行度
    p_dict = cal_p(pd.read_csv(path + r'/train.csv', sep='\t'))
    cluster_sum = item_meta_df.shape[1] - 1

    # embed, score = item_embed_pred["embedding"], item_embed_pred["prediction"]

    gru_pred_file = '../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method)
    # gru 预测数据
    pred_data = np.load(gru_pred_file)
    embed, sorted_idx, score = pred_data["embed"], pred_data['pred'], pred_data['score']
    xl, xu = 0, len(sorted_idx[0]) - 1

    hotcold_file = f"../../data/{dataset}/{dataset}_item_count.npz"
    # hot cold 数据
    # nov_data = np.load(hotcold_file)
    # nov_idx = nov_data['idxes']

    # ids_embed = pd.read_csv(r"../../data/ml-1m/ml-1m_ids_embed.csv", sep='\t')
    user_list = test_df['user_id'].values
    ground_truth = test_df['item_id'].values.reshape((-1, 1))
    neg_items = str_toarray(test_df['neg_items'].values)
    item_list = np.concatenate((ground_truth, neg_items), axis=1)

    n_var = [5, 10]

    # 计算聚类中心的pop
    # 读入聚类中心数据
    res_dict = {}
    cluster_center_df = pd.read_csv(path + "/cluster_center_info_vec_2.csv", sep='\t')
    print("聚类中心数据已读取")
    center_user_list = cluster_center_df['user_id'].values
    #
    for k in n_var:
        filename = f'../res_{dataset}/{dataset}_{method}_moea_{k}_center_vec_2.csv'
        for i, uid in enumerate(center_user_list):
            print(f"k = {k} 第{i}个聚类进行计算pop")
            items = []
            pre_for_ith_user = sorted_idx[i]
            for j, idx in enumerate(pre_for_ith_user):
                # 真实的编号在数据集中的编号
                true_num = item_list[i, idx]  # 意味着，在i-th用户数据上的第idx个物体，在真实数据集中是第true_num号物体
                # nov_score = np.where(nov_idx == true_num)[0][0]
                items.append(Item(idx=idx, true_num=true_num, acc_score=j, embed=embed[true_num]))

            pop, fit, seq = solve(uid, items, True, k=k, xl=xl,
                                  xu=xu)  # pop中的每一个individual，如[5,2,0,1,3]代表test_items中第i行的第[5,2,0,1,3]个物品
            s_id = select_ind(pop)
            res_dict = {'label': [i], 'pop': [pop],
                        'fit': [fit], 'sid': s_id}
            res_df = pd.DataFrame(res_dict)
            if os.path.exists(filename):  # 已存在追加
                res_df.to_csv(filename, mode='a', header=False, index=0, sep='\t')
            else:
                res_df.to_csv(filename, index=0, sep='\t')