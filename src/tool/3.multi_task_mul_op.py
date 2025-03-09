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
import numpy as np
from pymoo.core.repair import Repair

import copy


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


def ini_pop(popsize=100):
    uid_list = []
    pop_list = []
    label_list = []
    fit_list = []
    ini_dict = {'user_id': uid_list, 'label': label_list, 'pop': pop_list, 'fit': fit_list}
    # 得到每个用户的用户标签
    # all_user_label_df = pd.read_csv(path + "/cluster_label_train1_info.csv", sep='\t')
    all_user_label_df = pd.read_csv(path + "/cluster_label_train1_info_vec_2.csv", sep='\t')
    # 所有聚类标签
    # all_label = all_user_label_df['diversity_cluster_label'].values.tolist()
    all_label = all_user_label_df['cluster_label'].values.tolist()
    # 按聚类结果分成种群
    for c in range(len(set(all_label))):
        # 当前种群
        # 对每个用户进行初始化
        # cur_user = all_user_label_df.loc[all_user_label_df['diversity_cluster_label'] == c, 'user_id'].values.tolist()
        cur_user = all_user_label_df.loc[all_user_label_df['cluster_label'] == c, 'user_id'].values.tolist()
        for uid in cur_user:
            label = all_label[uid - 1]
            penc = all_user_label_df['penc'][uid - 1]

            choose_pop_len = int(popsize * penc)
            other_pop_len = popsize - choose_pop_len

            center_pop = ast.literal_eval(center_pop_info.loc[label, 'pop'])
            choose_pop = random.sample(center_pop, choose_pop_len)

            pool_other_pop = []
            for idx in range(len(center_pop_info.iloc[:, 0])):
                if idx != label:
                    temp = ast.literal_eval(center_pop_info.loc[idx, 'pop'])
                    for t in temp:
                        pool_other_pop.append(t)

            other_pop = random.sample(pool_other_pop, other_pop_len)

            init_p = choose_pop + other_pop
            uid_list.append(uid)
            pop_list.append(init_p)
            label_list.append(label)
            problem = MyProblem(uid, items, xl, xu, k)
            fit = []
            for te in init_p:
                fit.append(problem.evaluate(te)[0])
            fit_list.append(fit)
    ini_res = pd.DataFrame(ini_dict).sort_values(by='user_id')
    return ini_res


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


def change(off1, off2, i):
    temp1 = off1[i]
    temp2 = off2[i]
    if temp1 in off2:
        off1[i] = temp2
    if temp2 in off1:
        off2[i] = temp1
    if temp1 not in off2 and temp2 not in off1:
        off1[i] = temp2
        off2[i] = temp1
    return off1, off2


def cross_same(par1, par2, xl, xu, prob=0.9):
    prb = random.random()
    off1, off2 = copy.deepcopy(par1), copy.deepcopy(par2)
    if prb > prob:
        # 随机选择三个索引
        idx_list = random.sample(range(len(off1)), 3)
        # 将三个索引上的item进行交换
        for i in idx_list:
            off1, off2 = change(off1, off2, i)
            # 确保变异后的x在范围[xl, xu]内
            off1[i] = np.clip(off1[i], xl, xu)
            off2[i] = np.clip(off2[i], xl, xu)
    return off1, off2


def cross_dif(par1, par2, xl, xu, prob=0.9, eta=2.0):
    off1, off2 = copy.deepcopy(par1), copy.deepcopy(par2)
    if np.random.rand() <= prob:
        for i in range(len(par1)):
            if np.random.rand() <= 0.5:
                if par2[i] < par1[i]:
                    par1[i], par2[i] = par2[i], par1[i]
                # 计算beta
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                # 生成后代
                off1[i] = int(0.5 * ((1 + beta) * par1[i] + (1 - beta) * par2[i]))
                off2[i] = int(0.5 * ((1 - beta) * par1[i] + (1 + beta) * par2[i]))
                # 确保变异后的x在范围[xl, xu]内
                off1[i] = np.clip(off1[i], xl, xu)
                off2[i] = np.clip(off2[i], xl, xu)
    return off1, off2


def mutation(off, xl, xu, mut_prob=0.2, eta_mut=20):
    for i in range(len(off)):
        # 如果随机数小于变异概率，则进行变异
        if np.random.rand() < mut_prob:
            delta_q = 0

            # 生成[0,1]之间的随机数
            rnd = np.random.rand()

            if rnd <= 0.5:
                delta_q = ((2 * rnd) ** (1 / (eta_mut + 1))) - 1
            else:
                delta_q = 1 - ((2 * (1 - rnd)) ** (1 / (eta_mut + 1)))

            off[i] = off[i] + delta_q * (xu - xl)

            # 确保变异后的x在范围[xl, xu]内
            off[i] = int(np.clip(off[i], xl, xu))

    return off


def is_break(hist_pop, pop):
    hist_p = hist_pop['pop'].values.tolist()
    temp_p = pop['pop'].values.tolist()
    for idx in range(len(hist_p)):
        hist_cur = np.array(hist_p[idx])
        temp_cur = np.array(temp_p[idx])
        if (hist_cur != temp_cur).any():
            return False
    return True


from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
def enviromentSelection_elitism(pop):
    # pop: 格式df:{'user_id': uid_list, 'label': label_list, 'pop': pop_list, 'fit': fit_list}
    uid_len = pop['user_id'].values.tolist()
    pop_list = pop['pop'].values.tolist()
    fit_list = pop['fit'].values.tolist()
    survival = RankAndCrowdingSurvival()
    for idx, uid in enumerate(uid_len):
        # print(f"第{uid}个用户在进行环境选择")
        cur_pop = pop_list[idx]
        cur_fit = fit_list[idx]
        problem = MyProblem(uid, items, xl, xu, k)
        num_of_objectives = len(cur_fit[0])
        cur_P = Population.new("X", np.array(cur_pop))
        cur_P.set("F", np.array(cur_fit).reshape(-1, num_of_objectives))
        survival_pop = survival.do(problem, cur_P, n_survive=100)
        pop_list[idx] = [ind.X for ind in survival_pop]
        fit_list[idx] = [ind.F for ind in survival_pop]
    pop['pop'] = pop_list
    pop['fit'] = fit_list
    return pop


def pair_cross_mut(pop):
    # print("2.1开始交叉")
    # 随机选择用户序号
    ran1 = random.sample(cur_pop['user_id'].values.tolist(), 1)[0]
    ran2 = random.sample(cur_pop['user_id'].values.tolist(), 1)[0]

    temp1 = cur_pop.loc[cur_pop["user_id"] == ran1]['pop'].values[0]
    temp2 = cur_pop.loc[cur_pop["user_id"] == ran2]['pop'].values[0]

    idx1, idx2 = 0, 0
    # 同一用户
    if ran1 == ran2:
        # 随机选择两个不同pop
        idx1, idx2 = random.sample(range(100), 2)
        par1, par2 = temp1[idx1], temp2[idx2]
        off1, off2 = cross_same(par1, par2, xl, xu)
    else:
        # 随机选择两个pop
        idx1 = random.randint(0, 99)
        idx2 = random.randint(0, 99)
        par1, par2 = temp1[idx1], temp2[idx2]
        off1, off2 = cross_dif(par1, par2, xl, xu)
    # print("2.2开始变异")
    # 变异
    off1 = mutation(off1, xl=xl, xu=xu)
    off2 = mutation(off2, xl=xl, xu=xu)

    # 添加到正确位置上
    temp1[idx1] = off1
    temp2[idx2] = off2
    fit1 = cur_pop.loc[cur_pop["user_id"] == ran1]['fit'].values[0]
    fit2 = cur_pop.loc[cur_pop["user_id"] == ran1]['fit'].values[0]

    pop.at[pop[pop['user_id'] == ran1].index[0], 'pop'] = temp1
    pop.at[pop[pop['user_id'] == ran2].index[0], 'pop'] = temp2

    p1 = MyProblem(ran1, items, xl, xu, k)
    p2 = MyProblem(ran2, items, xl, xu, k)
    te1 = p1.evaluate(off1)
    te2 = p2.evaluate(off1)
    if len(te1) == 1:
        fit1[idx1] = p1.evaluate(off1)[0]
    else:
        fit1[idx1] = p1.evaluate(off1)
    if len(te2) == 1:
        fit2[idx2] = p2.evaluate(off2)[0]
    else:
        fit2[idx2] = p2.evaluate(off2)

    pop.at[pop[pop['user_id'] == ran1].index[0], 'fit'] = fit1
    pop.at[pop[pop['user_id'] == ran2].index[0], 'fit'] = fit2

    return pop


if __name__ == '__main__':
    np.random.seed(42)

    # 读入数据集
    # embed, score = item_embed_pred["embedding"], item_embed_pred["prediction"]
    dataset = 'ml-1m'   # 'ml-1m' 'Grocery_and_Gourmet_Food'
    method = 'ContraRec'  # 'GRU4Rec' 'ComiRec' 'SLRCPlus' 'SASRec' 'ContraRec' 'TiSASRec' 'KDA' 'Caser' 'NARM'
    path = r'../../data/' + dataset
    dev_df = pd.read_csv(path + r'/dev.csv', sep='\t')
    test_df = pd.read_csv(path + r'/test.csv', sep='\t')
    item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')
    train_df = pd.read_csv(path + r'/train1.csv', sep='\t')

    cluster_sum = item_meta_df.shape[1] - 1
    gru_pred_file = '../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method)
    # gru 预测数据
    pred_data = np.load(gru_pred_file)
    embed, sorted_idx, score = pred_data["embed"], pred_data['pred'], pred_data['score']
    xl, xu = 0, len(sorted_idx[0]) - 1

    # hotcold_file = f"../../data/{dataset}/{dataset}_item_count.npz"
    # # hot cold 数据
    # nov_data = np.load(hotcold_file)
    # nov_idx = nov_data['idxes']

    # ids_embed = pd.read_csv(r"../../data/ml-1m/ml-1m_ids_embed.csv", sep='\t')
    user_list = test_df['user_id'].values
    ground_truth = test_df['item_id'].values.reshape((-1, 1))
    neg_items = str_toarray(test_df['neg_items'].values)
    item_list = np.concatenate((ground_truth, neg_items), axis=1)

    # center_train_df = pd.read_csv(path + r'/cluster_label_train1_info.csv', sep='\t')
    # center_user = []
    # for la in center_train_df.loc['label']:
    #     cur_user = center_train_df.loc[center_train_df.loc['diversity_cluster_label'] == la, 'user_id']
    #     center_user.append(cur_user)

    items = []
    for i, uid in enumerate(user_list):
        # print("第{}个用户".format(uid))
        pre_for_ith_user = sorted_idx[i]
        for j, idx in enumerate(pre_for_ith_user):
            # 真实的编号在数据集中的编号
            true_num = item_list[i, idx]  # 意味着，在i-th用户数据上的第idx个物体，在真实数据集中是第true_num号物体
            # nov_score = np.where(nov_idx == true_num)[0][0]
            items.append(Item(idx=idx, true_num=true_num, acc_score=j, embed=embed[true_num]))

    n_var = [5, 10]

    for k in n_var:
        center_pop_info = pd.read_csv(f'../res_{dataset}/{dataset}_{method}_moea_{k}_center_vec_2.csv', sep='\t')
        filename = '../res_{}/{}_{}_moea_{}_test_vec_2_10_1.csv'.format(dataset, dataset, method, k)
        start = time.time()
        # 为每个用户初始化种群
        print("1.开始初始化")
        pop = ini_pop()  # 格式df:{'user_id': uid_list, 'label': label_list, 'pop': pop_list, 'fit': fit_list}
        hist_pop = pop.copy()
        # 同一类中进行多目标优化
        print("2.开始分组多目标多任务优化")
        for idx in range(len(center_pop_info['label'])):
            cur_pop = pop.loc[pop['label'] == idx]
            gen = 1
            while(1):
                print(f"k={k} 第{idx}个组 第{gen}次迭代")
                # 随机选择本次迭代要选择多少对用户进行交叉变异
                ran_len = len(cur_pop['user_id'].values.tolist())
                # print(ran_len)
                ran_id = random.randint(1, ran_len)
                # if idx == 1:
                #     print(1)
                for cnt in range(ran_len):
                    pop = pair_cross_mut(pop)
                # print("2.3开始环境选择")
                # environmentSelection
                pop = enviromentSelection_elitism(pop)
                # 判断是否达到中止条件（连续两代未改变）或者迭代达到30次
                if is_break(hist_pop, pop) or gen == 10:
                    print("达到终止条件!")
                    break
                gen += 1
                hist_pop = pop.copy()

        # 保存
        pop.to_csv(filename, index=0, sep='\t')
        end = time.time()
        print(f"花费时间{end - start}")
        # 计算每一个用户的hyper
        # for uid in user_list:
        #     cur_fit = pop.loc[pop[pop['user_id'] == uid].index[0], 'fit']
        #     plot_hy(cur_fit)
