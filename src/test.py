import os
from tqdm import *
import pandas as pd
import numpy as np
from mmpc import Predictor

sym_nums = 11
syms = ["sym_"+str(i) for i in range(sym_nums)]

window_sizes = [5,10,20,40,60]
targets = ["predict_window_"+str(i) for i in window_sizes]

root_dir = '../../test/data'


# #判定样本合理性（不跨交易时段等）
# #Demonstration purpose only.
def sample_is_leagel(sample):
    return True

# #随机采样器
# #Demonstration purpose only. actual sampler not revealed.
# def MySampler(df):
#     for i in range(100, len(df)):
#         sample = df.iloc[i-100:i].copy()
#         if sample_is_leagel(sample):
#             yield i, sample

def count_predict(df):
    counts = []
    for t in window_sizes:
        temp_cnt = [[0,0,0],[0,0,0],[0,0,0]]
        label = f'label_{t}'
        p_label = f'p_label_{t}'
        for i in range(3):
            for j in range(3):
                temp_cnt[i][j] = ((df[p_label] == i) & (df[label] == j)).sum()
        counts.append(temp_cnt)
    return np.array(counts)


def calc_pnl(df):
    pnls=pd.DataFrame()
    for i,t in enumerate(window_sizes):
        df['p_label_{}'.format(t)]=df['p_label_{}'.format(t)].apply(lambda x:x-1)
        df['mp_change_{}'.format(t)]=(df['n_midprice'].shift(-t)-df['n_midprice'])/(1+df['n_midprice'])
        pnls[targets[i]]=(df['p_label_{}'.format(t)]*df['mp_change_{}'.format(t)]).fillna(0)
    return pnls





def evaluate(df):

    y_pred = np.full((len(df), 5), np.nan)

    # sampler = MySampler(df)
    for i in range(100, len(df)):
        # y_pred[i] = np.array(predictor.predict(df.iloc[i-100:i,:26].copy()))
        y_pred[i] = np.array(predictor.predict(df.iloc[i-100:i,:26]))

    df[['p_label_5','p_label_10','p_label_20','p_label_40','p_label_60']] = pd.DataFrame(y_pred)
    counts = count_predict(df)
    pnls = calc_pnl(df)
    return counts, pnls




def metric(tp,fp,fn,beta = 0.5):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = (1+(beta**2)) * (precision*recall) / ((beta**2) * precision + recall)
    return precision,recall,f_score

def calc_final_metric(counts_df,pnls_dfs):
    ## 计算指标
    final_result = []
    for target in targets:
        total_TP = 0
        total_FP = 0
        # total_TN = 0
        total_FN = 0
        total_PNL = 0
        for sym in syms:

            ##模型预测为上涨或下跌，且预测正确的样本数
            True_Positive = counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==0)]['true_label:0'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==2)]['true_label:2'].values

            ##模型预测为上涨或下跌，但预测错误的样本数
            False_Positive = counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==0)]['true_label:1'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==0)]['true_label:2'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==2)]['true_label:0'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==2)]['true_label:1'].values

            ##真实样本中标注为不变，且模型也预测为不变的样本数
            # True_Negtive = counts_df[(counts_df['sym']==syms) & (counts_df['target']==target) & (counts_df['predict_label']==1)]['true_label:1'].values

            ##真实样本中标注为上涨、下跌，但模型预测错误的样本数
            False_Negtive = counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==1)]['true_label:0'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==2)]['true_label:0'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==0)]['true_label:2'].values \
                            + counts_df[(counts_df['sym']==sym) & (counts_df['target']==target) & (counts_df['predict_label']==1)]['true_label:2'].values

            total_TP += True_Positive
            total_FP += False_Positive
            # total_TN += True_Negative
            total_FN += False_Negtive

            precision,recall,f_score = metric(True_Positive,False_Positive,False_Negtive)

            pnl = pnls_dfs[sym][target].sum()
            total_PNL += pnl

            final_result.append([sym,target,precision,recall,f_score,pnl])

        precision,recall,f_score = metric(total_TP,total_FP,total_FN)
        final_result.append(['total',target,precision,recall,f_score,total_PNL])

    return pd.DataFrame(final_result, columns=['sym','target','precision','recall','f_score','pnl'])

if __name__ == '__main__':



    predictor = Predictor()


    pnls_dfs = {}
    counts_df = pd.DataFrame()

    ## 逐个股票读取文件并推理
    for sym in tqdm(syms, desc="process symbols"):
        path = os.path.join(root_dir,sym)
        filenames = os.listdir(path)

        pnls_dfs[sym] = pd.DataFrame()
        total_counts = np.zeros((len(window_sizes),3,3))

        for filename in tqdm(filenames, desc="files"):
            # print(os.path.splitext(filename))
            if os.path.splitext(filename)[-1] == '.csv':
                # print(f'processing {filename}')
                df = pd.read_csv(os.path.join(path, filename))
                counts, pnls = evaluate(df)
                total_counts += counts
                pnls_dfs[sym] = pnls_dfs[sym].append(pnls)

         ## 转化count的numpy matrix为dataframe
        for j,target in enumerate(targets):
            df = pd.DataFrame(total_counts[j],columns=['true_label:0','true_label:1','true_label:2'])
            df['sym'] = sym
            df['target'] = target
            df['predict_label'] = [0,1,2]
            counts_df = counts_df.append(df)



    ## 输出counts
    col_list = ['sym','target','predict_label','true_label:0','true_label:1','true_label:2']
    counts_df[col_list].to_csv('label_counts.csv',index=False)

    ## 输出详细pnls
    for sym in syms:
        pnls_dfs[sym].to_csv(f'pnls_{sym}.csv',index=False)

    final_df = calc_final_metric(counts_df,pnls_dfs)
    final_df.to_csv('result.csv',index=False)
