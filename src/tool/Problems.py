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
from multiprocessing import Pool
from pymoo.config import Config
from sklearn.metrics.pairwise import cosine_similarity

Config.show_compile_hint = False

# 忽略除零警告
np.seterr(divide='ignore', invalid='ignore')

def str_toarray(sitems):
    res = []
    for st in sitems:
        tmp = st.replace('[', '').replace(']', '').split(',')
        intt = [int(t) for t in tmp]
        res.append(intt)
    return np.array(res)

cluster_sum = 18
dataset = 'ml-1m'  # 'ml-1m' 'Office_Products' 'Clothing_Shoes_and_Jewelry' 'Grocery_and_Gourmet_Food'
method = 'GRU4Rec'  # 'GRU4Rec' 'ComiRec' 'SLRCPlus' 'SASRec'
path = r'../../data/' + dataset
dev_df = pd.read_csv(path + r'/dev.csv', sep='\t')
test_df = pd.read_csv(path + r'/test.csv', sep='\t')
item_meta_df = pd.read_csv(path + r'/item_meta.csv', sep='\t')
train_df = pd.read_csv(path + r'/train.csv', sep='\t')

gru_pred_file = '../res_{}/{}_{}_pred.npz'.format(dataset, dataset, method)
# gru 预测数据
pred_data = np.load(gru_pred_file)
embed, sorted_idx, score = pred_data["embed"], pred_data['pred'], pred_data['score']
xl, xu = 0, len(sorted_idx[0]) - 1

user_list = test_df['user_id'].values
ground_truth = test_df['item_id'].values.reshape((-1, 1))
neg_items = str_toarray(test_df['neg_items'].values)
item_list = np.concatenate((ground_truth, neg_items), axis=1)


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


# 计算项目流行度
def cal_p(df):
    out = {}
    # print(item_meta_df.columns)
    for i in range(1, len(item_meta_df['item_id']) + 1):
        temp = df[df['item_id'] == i]
        if i not in out.keys():
            out[i] = len(temp['item_id'])
    return out


p_dict = cal_p(pd.read_csv(path + r'/train.csv', sep='\t'))


def cal_novel(real_rec_ind):
    p = 0
    for i in real_rec_ind:
        if p_dict[i] == 0:
            p += 1
        else:
            p += 1 / p_dict[i]
    return p / len(real_rec_ind)


# 多目标优化
# 定义优化问题
class MyProblem(Problem):
    def __init__(self, user_id, items, xl, xu, k=6):
        super().__init__(n_var=k, n_obj=3, n_constr=0, xl=xl, xu=xu)
        self.items = items
        self.user_id = user_id

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for xi in x:
            acc = 0
            real_rec_ind = item_list[self.user_id - 1][xi]
            # 计算diversity和novelty
            diversity = cal_div(real_rec_ind)
            novelty = cal_novel(real_rec_ind)
            for item_id in xi:
                # acc
                acc += self.items[item_id].acc_score
            acc = acc / len(xi)
            F.append([-diversity, -novelty, acc])
        out["F"] = np.array(F)
        out["G"] = None


class MyProblem1(Problem):
    def __init__(self, user_id, items, xl, xu, k=6):
        super().__init__(n_var=k, n_obj=2, n_constr=0, xl=xl, xu=xu)
        self.items = items
        self.user_id = user_id

    def _evaluate(self, x, out, *args, **kwargs):
        F = []
        for xi in x:
            acc = 0
            real_rec_ind = item_list[self.user_id - 1][xi]
            # # 计算item间的diversity
            for item_id in xi:
                # acc
                acc += self.items[item_id].acc_score
            acc = acc / len(xi)
            embeds = np.array([self.items[j].embed for j in xi])
            cosine_dis = cosine_similarity(embeds, embeds)
            obj2 = (np.sum(cosine_dis) - len(xi)) / 2
            F.append([acc, obj2])
        out["F"] = np.array(F)
        out["G"] = None


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
                    while np.any(np.isin(xs[j], ctmp)):
                        xs[j] = np.random.randint(problem.xl, problem.xu, 1)[0]
                    ctmp.append(xs[j])
            return pop
        else:
            n = len(pop.X)
            xs = pop.X
            ctmp = []
            for j in range(n):
                # 保证个体内物体不能重复
                while np.any(np.isin(xs[j], ctmp)):
                    xs[j] = np.random.randint(problem.xl, problem.xu, 1)[0]
                ctmp.append(xs[j])
            return pop


def get_alg(pop, uid, items, popsize=100, k=5, xl=1, xu=99, gen=10):

    algorithm = AGEMOEA(pop_size=popsize,
                        sampling=pop,
                        crossover=get_crossover("int_sbx", eta=2, prob=0.9),
                        mutation=get_mutation("int_pm", prob=0.2, eta=20),
                        eliminate_duplicates=False,
                        repair=MyRepair()
                        )
    problem = MyProblem(items=items, user_id=uid, k=k, xl=xl, xu=xu)
    res = minimize(
        problem,
        algorithm,
        ('n_gen', gen)
    )
    pop = []
    for r in res.algorithm.pop.get("X"):
        tmp = []
        for rr in r:
            t = int(rr)
            tmp.append(items[t].idx)
        pop.append(tmp)

    # fit = res.algorithm.pop.get("F")
    fit = []
    for f in res.algorithm.pop.get("F"):
        temp = []
        for t in f:
            temp.append(t)
        fit.append(temp)

    return pop, fit
