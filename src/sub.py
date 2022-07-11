from cmath import log
import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
import random
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

torch.manual_seed(1024)
np.random.seed(1024)
random.seed(1024)

def feature_engineering():
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
                    'relatime',# 'sym',
                    'spread1','mid_price1',
                    'spread2','mid_price2',
                    'spread3','mid_price3',
                    'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                    'vol1_rel_diff','volall_rel_diff','label_5','label_10','label_20','label_40','label_60', 
                ]
    dataTensor=torch.zeros(10,79,2,1999,39,dtype=torch.float)
    for sym in range(10):
        for date in range(79):
            for timeIndex, time in enumerate(['am', 'pm']):  
                filePath = f"../data/raw/snapshot_sym{sym}_date{date}_{time}.csv"
                if not os.path.isfile(filePath):
                    print(filePath)
                    continue
                new_df = pd.read_csv(filePath)

                # region feature engineering
                # 价格+1（从涨跌幅还原到对前收盘价的比例）
                new_df['bid1'] = new_df['n_bid1']+1
                new_df['bid2'] = new_df['n_bid2']+1
                new_df['bid3'] = new_df['n_bid3']+1
                new_df['bid4'] = new_df['n_bid4']+1
                new_df['bid5'] = new_df['n_bid5']+1
                new_df['ask1'] = new_df['n_ask1']+1
                new_df['ask2'] = new_df['n_ask2']+1
                new_df['ask3'] = new_df['n_ask3']+1
                new_df['ask4'] = new_df['n_ask4']+1
                new_df['ask5'] = new_df['n_ask5']+1
        
                # 量价组合
                new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
                new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
                new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
                new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
                new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
                new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
                new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
                new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])

                new_df['relative_spread1'] = new_df['spread1'] / new_df['mid_price1']
                new_df['relative_spread2'] = new_df['spread2'] / new_df['mid_price2']
                new_df['relative_spread3'] = new_df['spread3'] / new_df['mid_price3']

                # 时间特征
                new_df['relatime'] = pd.to_datetime(new_df["time"]).map(lambda x: ((x.hour-9)+(x.minute*60+x.second)/3600))
                
                # 对量取对数
                new_df['bsize1'] = (new_df['n_bsize1']*10000).map(np.log1p)
                new_df['bsize2'] = (new_df['n_bsize2']*10000).map(np.log1p)
                new_df['bsize3'] = (new_df['n_bsize3']*10000).map(np.log1p)
                new_df['bsize4'] = (new_df['n_bsize4']*10000).map(np.log1p)
                new_df['bsize5'] = (new_df['n_bsize5']*10000).map(np.log1p)
                new_df['asize1'] = (new_df['n_asize1']*10000).map(np.log1p)
                new_df['asize2'] = (new_df['n_asize2']*10000).map(np.log1p)
                new_df['asize3'] = (new_df['n_asize3']*10000).map(np.log1p)
                new_df['asize4'] = (new_df['n_asize4']*10000).map(np.log1p)
                new_df['asize5'] = (new_df['n_asize5']*10000).map(np.log1p)
                new_df['amount'] = (new_df['amount_delta']/100000).map(np.log1p)
                
                new_df['vol1_rel_diff']   = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['volall_rel_diff'] = (new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                - new_df['n_asize1'] - new_df['n_asize2'] - new_df['n_asize3'] - new_df['n_asize4'] - new_df['n_asize5'] ) / \
                                ( new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                + new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5'] )
                
                # endregion
                dataTensor[sym][date][timeIndex]=torch.tensor(new_df[columns_need].to_numpy(),dtype=torch.float)
    pickle.dump(dataTensor,open('../data/pkl/all.pkl','wb'))
    return dataTensor

class Dataset(data.Dataset):
    def __init__(self, _dataTensor, labelIndex):
        self._dataTensor = _dataTensor
        self.labelIndex = labelIndex
        self.length = 0
        self.indexToSym=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToDate=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToTime=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        self.indexToStartIndex=torch.zeros(_dataTensor.nelement(),dtype=torch.int)
        for sym in range(self._dataTensor.shape[0]):
            for date in range(self._dataTensor.shape[1]):
                for time in range(self._dataTensor.shape[2]):
                    if torch.sum(self._dataTensor[sym][date][time]).item() != 0:
                        for startIndex in range(1900):
                            self.indexToSym[self.length]=sym
                            self.indexToDate[self.length]=date
                            self.indexToTime[self.length]=time
                            self.indexToStartIndex[self.length]=startIndex
                            self.length+=1
                    # else:
                    #     print(sym,date,time)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        sym = self.indexToSym[index]
        date = self.indexToDate[index]
        time = self.indexToTime[index]
        startIndex = self.indexToStartIndex[index]
        data = self._dataTensor[sym][date][time][startIndex:startIndex+100,:-5]
        label = self._dataTensor[sym][date][time][startIndex+99][-5+self.labelIndex]
        return data,label.to(torch.long)

def construct_dataset(batchSize,sym,label_index):
    if os.path.exists('../data/pkl/all.pkl'):
        dataTensor = pickle.load(open('../data/pkl/all.pkl','rb'))
    else:
        dataTensor=feature_engineering()
    dataTensor=dataTensor.to('cuda')
    # print(dataTensor.device)
    labelIndex=label_index # 0 for label_5, 1 for label_10, 2 for label_20, 3 for label_40, 4 for label_60
    trainTensor=dataTensor[sym:sym+1,:63]
    validateTensor=dataTensor[sym:sym+1,63:71]
    testTensor=dataTensor[sym:sym+1,71:]
    # print(trainTensor.shape)
    # print(validateTensor.shape)
    # print(testTensor.shape)
    # fdshkj
    trainDataset = Dataset(_dataTensor=trainTensor,labelIndex=labelIndex)
    validateDataset=Dataset(_dataTensor=validateTensor,labelIndex=labelIndex)
    testDataset=Dataset(_dataTensor=testTensor,labelIndex=labelIndex)
    # print(testDataset)
    trainLoader = data.DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    validateLoader=data.DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=True)
    testLoader=data.DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=True)
    # print(testLoader.)
    return trainLoader,validateLoader,testLoader

class MLP(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.net = nn.Linear(3200, num_classes)
        # self.net = nn.Sequential(nn.Linear(3200, 64),nn.ReLU(),nn.Linear(64, num_classes))
        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.uniform_(m.weight)
        self.net.apply(init_weight)
    def forward(self, x):        
        x=torch.flatten(x,start_dim=1)
        x=self.net(x)
        x=torch.softmax(x,dim=1)
        return x

class LSTM(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=33, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 64),nn.ReLU(),nn.Linear(64, num_classes))
    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = torch.softmax(x, dim=1)
        return x

class GRU(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.gru = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x,_ = self.gru(x)
        x = self.fc(x[:, -1, :])
        x = torch.softmax(x, dim=1)
        return x

class DeepLob(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,8)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1),stride=(2,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )        
        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        # lstm layers
        # self.lstm = nn.LSTM(input_size=48, hidden_size=64, num_layers=1, batch_first=True)
        # self.fc1 = nn.Linear(64, self.num_classes)
        self.fc = nn.Sequential(nn.Linear(384, 64),nn.ReLU(),nn.Linear(64, self.num_classes))

    def forward(self, x):
        x=torch.unsqueeze(x,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # lstm head
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        # x=torch.squeeze(torch.transpose(x,1,2))
        # x, _ = self.lstm(x)
        # x = self.fc1(x[:, -1, :])

        # MLP head
        x = x.reshape(-1,48*8)
        x = self.fc(x)

        x = torch.softmax(x, dim=1)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):
    
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        # transformer_model = nn.Transformer(d_model=32,nhead=8, num_encoder_layers=6,num_decoder_layers=6, 
            # dim_feedforward=64, dropout=0.1,batch_first=True)
        # src = torch.rand((10, 32, 512))
        # tgt = torch.rand((20, 32, 512))
        # out = transformer_model(src, tgt)
        # transformer_model.encoder
        # transformer_model.decoder
        # self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



def train_one_epoch(model:nn.Module, trainLoader:data.DataLoader, criterion, optimizer, scheduler):
    model.train()
    TP=0;TN=0;FP=0;FN=0
    rightCount=0;totalCount=0
    losses=torch.zeros(len(trainLoader),dtype=torch.float)
    for index, (input, label) in enumerate(trainLoader):
    # for index,(input, label) in enumerate(tqdm(trainLoader)):
        # print(label.shape[0])
        # print(torch.sum(label==0))
        # print(torch.sum(label==1))
        # print(torch.sum(label==2))
        output = model(input)
        label_0_weight=1/torch.sqrt(torch.sum(label==0)+1)
        label_1_weight=1/torch.sqrt(torch.sum(label==1)+1)
        label_2_weight=1/torch.sqrt(torch.sum(label==2)+1)
        # label_0_weight=1/(torch.sum(label==0)+1)
        # label_1_weight=1/(torch.sum(label==1)+1)
        # label_2_weight=1/(torch.sum(label==2)+1)
        total_weight=label_0_weight+label_1_weight+label_2_weight
        label_0_weight=label_0_weight/total_weight
        label_1_weight=label_1_weight/total_weight
        label_2_weight=label_2_weight/total_weight
        weight=torch.tensor([label_0_weight,label_1_weight,label_2_weight]).to('cuda')
        # print(weight)
        criterion=nn.CrossEntropyLoss(weight=weight)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[index]=loss

        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))
        rightCount+=torch.sum(predictLabel==label)
        totalCount+=label.shape[0]
        # writer.add_scalars('DeepLob',{'trainScore':trainScore,'validateScore':validateScore,"trainLoss":trainLoss,"validateLoss":validateLoss}, epoch)
    scheduler.step()
    if TP==0:
        f_beta=torch.tensor([0],dtype=torch.float32,device='cuda')
    else:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        beta=0.5
        f_beta=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    accuracy=rightCount/totalCount
    return (torch.sum(losses)/losses.nelement()).item(),f_beta.item(),accuracy.item()

def validate_one_epoch(model:nn.Module, validateLoader:data.DataLoader):
    model.eval()
    TP=0;TN=0;FP=0;FN=0
    rightCount=0;totalCount=0
    for (input, label) in validateLoader:
    # for (input, label) in tqdm(validateLoader):
        output = model(input)
        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))
        rightCount+=torch.sum(predictLabel==label)
        totalCount+=label.shape[0]
    if TP==0:
        f_beta=torch.tensor([0],dtype=torch.float32,device='cuda')
    else:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        beta=0.5
        f_beta=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    accuracy=rightCount/totalCount
    return f_beta.item(),accuracy.item()

def trial(model:nn.Module,modelName,epochs,batchSize,savedName,sym,label_index):
    trainLoader,validateLoader,testLoader=construct_dataset(batchSize=batchSize,sym=sym,label_index=label_index)

    model.to('cuda')
    # summary(model, (512, 100, 32))
    
    # criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.42,0.14,0.42],dtype=torch.float32).to('cuda'))
    criterion=nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=5,eta_min=1e-6)

    maxBearableEpochs=30
    noProgressEpochs=0
    stopEpoch=0
    currentBestScore=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainLoss,trainScore,trainAccuracy=train_one_epoch(model=model, trainLoader=trainLoader, criterion=criterion,optimizer=optimizer,scheduler=scheduler)
        validateScore,validateAccuracy=validate_one_epoch(model=model, validateLoader=validateLoader)
        print("epoch:",epoch,"trainLoss:",trainLoss,"trainScore:",trainScore,"trainAccuracy:",trainAccuracy,\
            "validateScore:",validateScore,"validateAccuracy",validateAccuracy,"delta:",validateScore-currentBestScore)
        writer.add_scalars(modelName,{'trainScore':trainScore,'validateScore':validateScore,"trainLoss":trainLoss,}, epoch)
        if validateScore > currentBestScore:
            currentBestScore=validateScore
            currentBestEpoch=epoch
            torch.save(model, "../weight/submit/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    testScore,testAccuracy=validate_one_epoch(model,validateLoader=testLoader)
    print("==========================================================================================")
    print("testScore",testScore,"validateScore",currentBestScore,"bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")

def test():
    mean_score_list=[0 for i in range(5)]
    for sym in range(10):
        for label_index in tqdm(range(5)):
            trainLoader,validateLoader,testLoader=construct_dataset(batchSize=512,sym=sym,label_index=label_index)
            model=torch.load("../weight/submit/"+str(sym)+"_"+str(label_index)+".pth")
            trainScore,trainAccuracy=validate_one_epoch(model,validateLoader=trainLoader)
            validateScore,validateAccuracy=validate_one_epoch(model,validateLoader=validateLoader)
            testScore,testAccuracy=validate_one_epoch(model,validateLoader=testLoader)
            print("sym",sym,"label_index",label_index,"trainScore",trainScore,"validateScore",validateScore,"testScore",testScore)
            mean_score_list[label_index]+=testScore
    for i in range(5):
        print(i,mean_score_list[i]/10)

if __name__=='__main__':
    # model=MLP(num_classes=3)
    # trial(model=model,modelName="MLP",epochs=200,batchSize=512,savedName="3.pth")

    # model=LSTM(num_classes=3)
    # trial(model=model,modelName="LSTM",epochs=40,batchSize=512,savedName="9_4.pth",sym=9,label_index=4)

    # model=GRU(num_classes=3)
    # trial(model=model,modelName="GRU",epochs=200,batchSize=512,savedName="2.pth")

    # model=DeepLob(num_classes=3)
    # trial(model=model,modelName="DeepLob",epochs=200,batchSize=512,savedName="10_3.pth",sym=0,label_index=0)

    test()
    print("All is well!")
    writer.close()


