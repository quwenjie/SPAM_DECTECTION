from datapreprocess import readfile, calculate_wordid
import os
import torch
import numpy as np
class SPAM_Dataset(torch.utils.data.Dataset):
    def __init__(self,train):
        self.rawdata=[]
        self.labels=[]
        self.text=[]
        self.word2id = calculate_wordid()
        for root, dirs, files in os.walk('.\data'):
            # root 表示当前正在访问的文件夹路径
            # dirs 表示该文件夹下的子目录名list
            # files 表示该文件夹下的文件list
            for f in files:
                filename = os.path.join(root, f)
                if(train):
                    if root[-1]=='1' or root[-1]=='2':
                        continue
                else:
                    if root[-1]!='1' and root[-1]!='2':
                        continue
                subwordvec,str = readfile(filename)
                arr=np.zeros(12000)
                for w in subwordvec:
                    id=self.word2id.get(w,0)
                    if id:
                        arr[id]+=1
                #arr/=len(subwordvec)
                self.rawdata.append(arr)
                if(f[0:3]=='spm'):
                    self.labels.append(1)
                else:
                    self.labels.append(0)
                self.text.append(str)


    def __len__(self):
        return len(self.rawdata)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.reshape(torch.FloatTensor(self.rawdata[idx]),(1,-1)),torch.FloatTensor([self.labels[idx]]),self.text[idx]
