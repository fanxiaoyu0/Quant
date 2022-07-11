from typing import List
import pandas as pd
import numpy as np
import torch
from sub import LSTM

def feature_engineering(x: pd.DataFrame):
    columns_need = ['bid1','bsize1',
                    'bid2','bsize2',
                    'bid3','bsize3',
                    'bid4','bsize4',
                    'bid5','bsize5',
                    'ask1','asize1',
                    'ask2','asize2',
                    'ask3','asize3',
                    'ask4','asize4',
                    'ask5','asize5',
                    'relatime', #'sym',
                    'spread1','mid_price1',
                    'spread2','mid_price2',
                    'spread3','mid_price3',
                    'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                    'vol1_rel_diff','volall_rel_diff'
    ]

    # region feature engineering
    # 价格+1（从涨跌幅还原到对前收盘价的比例）
    x['bid1'] = x['n_bid1']+1
    x['bid2'] = x['n_bid2']+1
    x['bid3'] = x['n_bid3']+1
    x['bid4'] = x['n_bid4']+1
    x['bid5'] = x['n_bid5']+1
    x['ask1'] = x['n_ask1']+1
    x['ask2'] = x['n_ask2']+1
    x['ask3'] = x['n_ask3']+1
    x['ask4'] = x['n_ask4']+1
    x['ask5'] = x['n_ask5']+1

    # 量价组合
    x['spread1'] =  x['ask1'] - x['bid1']
    x['spread2'] =  x['ask2'] - x['bid2']
    x['spread3'] =  x['ask3'] - x['bid3']
    x['mid_price1'] =  x['ask1'] + x['bid1']
    x['mid_price2'] =  x['ask2'] + x['bid2']
    x['mid_price3'] =  x['ask3'] + x['bid3']
    x['weighted_ab1'] = (x['ask1'] * x['n_bsize1'] + x['bid1'] * x['n_asize1']) / (x['n_bsize1'] + x['n_asize1'])
    x['weighted_ab2'] = (x['ask2'] * x['n_bsize2'] + x['bid2'] * x['n_asize2']) / (x['n_bsize2'] + x['n_asize2'])
    x['weighted_ab3'] = (x['ask3'] * x['n_bsize3'] + x['bid3'] * x['n_asize3']) / (x['n_bsize3'] + x['n_asize3'])

    x['relative_spread1'] = x['spread1'] / x['mid_price1']
    x['relative_spread2'] = x['spread2'] / x['mid_price2']
    x['relative_spread3'] = x['spread3'] / x['mid_price3']

    # 时间特征
    x['relatime'] = pd.to_datetime(x["time"]).map(lambda y: ((y.hour-9)+(y.minute*60+y.second)/3600))
    
    # 对量取对数
    x['bsize1'] = (x['n_bsize1']*10000).map(np.log1p)
    x['bsize2'] = (x['n_bsize2']*10000).map(np.log1p)
    x['bsize3'] = (x['n_bsize3']*10000).map(np.log1p)
    x['bsize4'] = (x['n_bsize4']*10000).map(np.log1p)
    x['bsize5'] = (x['n_bsize5']*10000).map(np.log1p)
    x['asize1'] = (x['n_asize1']*10000).map(np.log1p)
    x['asize2'] = (x['n_asize2']*10000).map(np.log1p)
    x['asize3'] = (x['n_asize3']*10000).map(np.log1p)
    x['asize4'] = (x['n_asize4']*10000).map(np.log1p)
    x['asize5'] = (x['n_asize5']*10000).map(np.log1p)
    x['amount'] = (x['amount_delta']/100000).map(np.log1p)
    
    x['vol1_rel_diff']   = (x['n_bsize1'] - x['n_asize1']) / (x['n_bsize1'] + x['n_asize1'])
    x['volall_rel_diff'] = (x['n_bsize1'] + x['n_bsize2'] + x['n_bsize3'] + x['n_bsize4'] + x['n_bsize5'] \
                    - x['n_asize1'] - x['n_asize2'] - x['n_asize3'] - x['n_asize4'] - x['n_asize5'] ) / \
                    ( x['n_bsize1'] + x['n_bsize2'] + x['n_bsize3'] + x['n_bsize4'] + x['n_bsize5'] \
                    + x['n_asize1'] + x['n_asize2'] + x['n_asize3'] + x['n_asize4'] + x['n_asize5'] )
    print(x[columns_need])
    print(torch.tensor(x[columns_need].to_numpy(), dtype=torch.float).shape)

    return torch.tensor(x[columns_need].to_numpy(), dtype=torch.float).unsqueeze(0).to('cuda')

class Predictor():
    def __init__(self):
        pass
    def predict(self, x: pd.DataFrame) -> List[int]:
        input = feature_engineering(x)
        sym=x['sym'].iloc[0]
        label_list=[]
        for i in range(5):
            model=torch.load('../weight/submit/'+str(sym)+'_'+str(i)+'.pth')
            model.to('cuda')
            output=model(input)
            predictLabel = torch.argmax(output, axis=1)
            label_list.append(predictLabel.item())
        return label_list