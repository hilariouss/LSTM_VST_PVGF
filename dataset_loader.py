import numpy as np
import pandas as pd
import copy


class Data:
    def __init__(self,pv_chunk,pv_normalized_chunk,pv_chunk_long,pv_normalized_chunk_long, pv_label):
        self.pv_chunk = pv_chunk
        self.pv_normalized_chunk = pv_normalized_chunk
        self.pv_chunk_long = pv_chunk_long
        self.pv_normalized_chunk_long = pv_normalized_chunk_long
        self.pv_label = pv_label


    def getLinearShapeInput(self):
        return self.pv_chunk

    def get2DShapeInput(self):
        input = [p for p in self.pv_normalized_chunk]
        input = np.array(input)
        input = input.reshape(input.shape[0],input.shape[1],1)
        return input

    def get2DShapeInput_long(self):
        input = [p for p in self.pv_normalized_chunk_long]
        input = np.array(input)
        input = input.reshape(input.shape[0],input.shape[1],1)
        return input

class Dataset_loader:
    def __init__(self,pvdir,trainset_ratio = 0.7,duration_hour =6,duration_hour_long = 24*21, attList =[5,6,7,8,9], attList_long = [5,6,7,8,9]):
        self.duration = duration_hour
        self.duration_long = duration_hour_long

        self.pv = np.array(pd.read_csv(pvdir)).astype(float)
        self.attribute_list = attList
        self.attribute_list_long = attList_long

        self.num_attribute = len(self.attribute_list)
        self.num_attribute_long = len(self.attribute_list_long)

        self.pv_normalized = self.attwise_normalization(copy.deepcopy(self.pv)).astype(float)

        self.dataset = self.genDataset()

    def attwise_normalization(self,arr):
        print(np.max(arr,axis=0))
        arr = (arr - np.mean(arr,axis=0)) / ((np.max(arr,axis=0) - np.min(arr,axis=0)+0.0001))
        return arr



    def genDataset(self):
        dataset = []
        for i in range(int(self.duration_long),self.pv.shape[0]-1): #i = current +1
            pv_chunk = self.pv[i-int(self.duration): i,self.attribute_list]
            pv_normalized_chunk = self.pv_normalized[i-int(self.duration): i,self.attribute_list]
            pv_chunk_long = self.pv[i - int(self.duration_long): i,self.attribute_list_long]
            pv_normalized_chunk_long = self.pv_normalized[i - int(self.duration_long): i,self.attribute_list_long]

            pv_label = self.pv[i+1,4]

            if(pv_label != 0 and pv_chunk[-1,4] != 0):
                dataset.append(Data(pv_chunk,pv_normalized_chunk,pv_chunk_long,pv_normalized_chunk_long,pv_label))
        return dataset


    def getDataset(self,shuffle = False,seed = 10, batch_size=128):
        trainset = []
        testset = []

        self.dataset = self.dataset[:len(self.dataset) - len(self.dataset)%batch_size]
        if(shuffle == True):
            np.random.seed(seed)

            shuffle_chunksize = 2
            p = np.random.permutation(int(len(self.dataset) / shuffle_chunksize))
            p2 = []
            for p_ in p:
                for i in range(shuffle_chunksize):
                    p2.append(p_ * shuffle_chunksize + i)
            self.dataset = np.array(self.dataset)[p2]




        testset_size = 1500  -  1500%batch_size
        for e, d in enumerate(self.dataset):
            if (e < len(self.dataset) - (testset_size) ):
                trainset.append(d)
            else:
                testset.append(d)



        return trainset, testset



def module_test():
    dl = Dataset_loader(pvdir = "./data/pv_2015_2016_gy_processed.csv")
    pass

if __name__ == "__main__":
    module_test()


'''
    self.pv = self.bind_according_to_timestep_interval(self.pv_origin,self.timestep_interval)
  
    def bind_according_to_timestep_interval(self,data,timestep_interval):
        boundData = []
        for i in range(0,data.shape[0],timestep_interval):
            data[i,2:5] = np.sum(data[i:i+4,2:5],axis=0)
            boundData.append(data[i])
        return np.array(boundData)
'''
'''
    def genDataset(self):
        dataset = []
        for i in range(int(self.duration),self.pv.shape[0]-1): #i = current +1
            pv_chunk = self.pv[i-int(self.duration): i]
            pv_normalized_chunk = self.pv_normalized[i-int(self.duration): i]
            pv_label = self.pv[i+1,3]

            if(pv_label != 0):#and pv_label <1490):
                dataset.append(Data(pv_chunk,pv_normalized_chunk,pv_label))
        return dataset
    '''