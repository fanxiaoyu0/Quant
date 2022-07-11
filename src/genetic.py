import random
import numpy as np
import pdb
import pickle
from matplotlib import pyplot as plot
from tqdm import tqdm
import json
import pandas as pd
import os
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
from torch.utils import data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
np.random.seed(1024)
random.seed(1024)
torch.manual_seed(1024)

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
                    'spread1','mid_price1',
                    'spread2','mid_price2',
                    'spread3','mid_price3',
                    'weighted_ab1','weighted_ab2','weighted_ab3','amount',
                    'vol1_rel_diff','volall_rel_diff','label_5','label_10','label_20','label_40','label_60', 
                ]
    dataTensor=torch.zeros(1,79,2,1999,37,dtype=torch.float)
    for sym in [4,]:
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
                # if sym not in dataDict.keys():
                    # dataDict[sym]={}
                # if date not in dataDict[sym].keys():
                    # dataDict[sym][date]={}
                dataTensor[0][date][timeIndex]=torch.tensor(new_df[columns_need].to_numpy(),dtype=torch.float)
    # print(dataTensor.shape)
    pickle.dump(dataTensor,open('../data/pkl/1.pkl','wb'))
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
                    else:
                        print(sym,date,time)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        sym = self.indexToSym[index]
        date = self.indexToDate[index]
        time = self.indexToTime[index]
        startIndex = self.indexToStartIndex[index]
        data = self._dataTensor[sym][date][time][startIndex:startIndex+100,:-5]
        label = self._dataTensor[sym][date][time][startIndex+99][-5+self.labelIndex]
        # print(label)
        # print(data.shape)
        # print(label.shape)
        # fsdhjk
        return data,label.to(torch.long)

def construct_dataset(batchSize):
    if os.path.exists('../data/pkl/1.pkl'):
        dataTensor = pickle.load(open('../data/pkl/1.pkl','rb'))
    else:
        dataTensor=feature_engineering()
    dataTensor=dataTensor.to('cuda')
    # print(dataTensor.device)
    labelIndex=0 # 0 for label_5, 1 for label_10, 2 for label_20, 3 for label_40, 4 for label_60
    trainTensor=dataTensor[:,:63]
    validateTensor=dataTensor[:,63:71]
    testTensor=dataTensor[:,71:]
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
    def __init__(self,featureDim):
        super().__init__()
        self.net = nn.Linear(featureDim*100, 3)
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
    def __init__(self,featureDim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=featureDim, hidden_size=64, num_layers=1, batch_first=True)
        # self.fc = nn.Sequential(nn.Linear(64, 64),nn.ReLU(),nn.Linear(64, num_classes))
        self.fc = nn.Linear(64, 3)
    def forward(self, x):
        x,_ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = torch.softmax(x, dim=1)
        return x

class Agent:
    def __init__(self,featureDim):
        self.network=MLP(featureDim=featureDim).to('cuda')
        # self.network=LSTM(featureDim=featureDim).to('cuda')
        self.fitness=torch.tensor([0],dtype=torch.float32,device='cuda')

def generate_agents(populationSize,featureDim):
    return [Agent(featureDim=featureDim) for _ in range(populationSize)]

def calculate_fitness(agents,input,label):
    for agent in agents:
    # for agent in tqdm(agents):
        agent:Agent
        model=agent.network
        model.eval()
        # with torch.no_grad():
        TP=0;TN=0;FP=0;FN=0
        # for (input, label) in trainLoader:
        # for (input, label) in tqdm(validateLoader):
        output = model(input)
        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))
        if TP==0:
            agent.fitness=torch.tensor([0],dtype=torch.float32,device='cuda')
        else:
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            beta=0.5
            agent.fitness=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    return agents

def selection(agents):
    agents=sorted(agents, key=lambda agent: agent.fitness.item(),reverse=True)
    selectedAgentList=[]
    for index,agent in enumerate(agents):
        if random.random()<=(1-index/len(agents))**2:# and random.random()<=(1-index/len(agents)):
            selectedAgentList.append(agent)
    return selectedAgentList

def mutation(agents:list,populationSize,featureDim):
    geneMutationRate=0.001
    offSpring = []
    for _ in range(populationSize-len(agents)):
        parent:Agent=agents[random.randint(0,len(agents)-1)]
        child=Agent(featureDim=featureDim)
        parentStateDict=parent.network.state_dict()
        childStateDict=child.network.state_dict()
        for name, param in parentStateDict.items():
            transformedParam=param*torch.randn(param.shape).to('cuda')*(torch.rand(param.shape)<geneMutationRate).to('cuda')+param
            childStateDict[name].copy_(transformedParam)
        offSpring.append(child)
    agents.extend(offSpring)
    return agents 

# def mutation(agents:list,populationSize,featureDim):
#     geneMutationRate=0.1
#     offSpring = []
#     for index,agent in enumerate(agents):
#         if random.random()>=(1-(index+1)/len(agents))**2:
#             child=Agent(featureDim=featureDim)
#             parentStateDict=agent.network.state_dict()
#             childStateDict=child.network.state_dict()
#             for name, param in parentStateDict.items():
#                 transformedParam=param*torch.randn(param.shape,device='cuda')*(torch.rand(param.shape,device='cuda')<geneMutationRate)+param
#                 childStateDict[name].copy_(transformedParam)
#             offSpring.append(child)
#     agents.extend(offSpring)
#     alienList=[]
#     for _ in range(populationSize-len(agents)):
#         alien=Agent(featureDim=featureDim)
#         alienStateDict=alien.network.state_dict()
#         for name, param in alienStateDict.items():
#             transformedParam=torch.randn(param.shape,device='cuda')
#             alienStateDict[name].copy_(transformedParam)
#         alienList.append(alien)
#     agents.extend(alienList)
#     return agents 

def evaluate_agent(agent:Agent,testLoader):
    model=agent.network
    model.eval()
    TP=0;TN=0;FP=0;FN=0
    for (input, label) in testLoader:
    # for (input, label) in tqdm(validateLoader):
        output = model(input)
        predictLabel = torch.argmax(output, axis=1)
        TP+=(torch.sum(torch.mul(predictLabel==2,label==2))+torch.sum(torch.mul(predictLabel==0,label==0)))
        TN+=torch.sum(torch.mul(predictLabel==1,label==1))
        FP+=(torch.sum(torch.mul(predictLabel==2,label!=2))+torch.sum(torch.mul(predictLabel==0,label!=0)))
        FN+=(torch.sum(torch.mul(predictLabel!=2,label==2))+torch.sum(torch.mul(predictLabel!=0,label==0)))
    if TP==0:
        fitness=torch.tensor([0],dtype=torch.float32,device='cuda')
    else:
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        beta=0.5
        fitness=(1+beta**2)*precision*recall/(beta**2*precision+recall)
    return fitness.item()

def run_genetic_algorithm(modelName,modelVersion,batchSize):
    trainLoader,validateLoader,testLoader=construct_dataset(batchSize=batchSize)
    featureDim=32
    populationSize=256
    generations=1000
    agents=generate_agents(populationSize=populationSize,featureDim=featureDim)
    currentBsetFitness=0
    for i in range(generations):
        for (input, label) in tqdm(trainLoader):
            agents=calculate_fitness(agents,input=input,label=label)
            agents=selection(agents)
            agents=mutation(agents,populationSize,featureDim=featureDim)
            # if index%100==0:
        trianFitness=agents[0].fitness.item()
        validateFitness=evaluate_agent(agent=agents[0],testLoader=validateLoader)
        writer.add_scalars("fitness",{"train":trianFitness,"validate":validateFitness}, i)
        print('Generation',i,"trainFitness",trianFitness,"validateFitness",validateFitness)
        if trianFitness>currentBsetFitness:
            currentBsetFitness=trianFitness
            torch.save(agents[0].network.state_dict(),'../weight/'+modelName+'/'+modelVersion+'.pth')
    testFitness=evaluate_agent(agent=agents[0],testLoader=testLoader)
    print("testFitness",testFitness)

if __name__=="__main__":
    # run_genetic_algorithm(modelName='MLP',modelVersion='5',batchSize=512)
    run_genetic_algorithm(modelName='LSTM',modelVersion='8',batchSize=16384)
    print("All is well!")
    writer.close()
