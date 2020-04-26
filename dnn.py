import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataset import SPAM_Dataset
from sklearn.tree import export_graphviz
BATCH_SIZE=5
SMALL_PACK=15
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(12000, 3000),nn.ReLU(),
                            nn.Linear(3000,500),nn.ReLU(),
                            nn.Linear(500,50),nn.ReLU(),
                            nn.Linear(50,1),nn.Sigmoid())

    def forward(self, x):
        hx=self.classifier(x)
        return hx

def train(model, device, trainloader,optimizer, criterion):
    model.train()
    run_loss=0
    batchidx=0
    tot=0
    for data,target,raw in trainloader:
        data, target = data.to(device), target.to(device)
        tot+=data.shape[0]
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        run_loss+=loss
        loss.backward()
        optimizer.step()
        if batchidx % SMALL_PACK == SMALL_PACK-1:
            print('Train :({:.0f}%)]\tLoss: {:.6f}'.format(
                tot/len(trainloader)
                , run_loss/SMALL_PACK))
            run_loss=0
        batchidx+=1
    scheduler.step()

def test( model, device,testloader):
    model.eval()
    correct = 0
    tot=0
    with torch.no_grad():
        for (data,target,raw) in testloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            tot+=data.shape[0]
            output = model(data)
            output=F.threshold(output,0.5,0)
            output=-F.threshold(-output,-0.5,-1)
            correct += target.eq(output.view_as(target)).sum().item()
    print(100.*correct/tot)
    #torch.save(model,'%d.pkl'%correct)
dic=[]
def decisiontree(x_train, x_test, y_train, y_test):

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    predict_results = clf.predict(x_test)
    global  dic
    print(dic)
    print('%.4f' % accuracy_score( clf.predict(x_train), y_train))
    print('%.4f' % accuracy_score(predict_results, y_test))
    export_graphviz(clf,  feature_names= dic,class_names=['normal',"spam"],out_file='.\\tree.dot')

    import graphviz
    graph = graphviz.Source('.\\tree.dot')
def LRP(name,data,target):
    net = torch.load(name)
    W = []
    B = []
    for i in range(0, 8, 2):
        W.append(net.classifier[i].weight.t().detach().cpu().numpy())
        B.append(net.classifier[i].bias.detach().cpu().numpy())
    L = len(W)
    X = data.cpu().numpy()
    T = target.cpu().numpy()
    A = [X] + [None] * L
    for l in range(L):
        A[l + 1] = A[l].dot(W[l]) + B[l]
        if (l + 1 != L):
            A[l + 1] = np.maximum(0, A[l + 1])
    R = [None] * L + [A[L]]
    for l in range(1, L)[::-1]:
        z = A[l].dot(W[l]) + B[l]
        s = R[l + 1] / z  # step 2
        c = s.dot(W[l].T)  # step 3
        R[l] = A[l] * c  # step 4
    w = W[0]
    wp = np.maximum(0, w)
    wm = np.minimum(0, w)
    lb = A[0] * 0 - 1
    hb = A[0] * 0 + 1

    z = A[0].dot(w) - lb.dot(wp) - hb.dot(wm) + 1e-9  # step 1
    s = R[1] / z  # step 2
    c, cp, cm = s.dot(w.T), s.dot(wp.T), s.dot(wm.T)  # step 3
    R[0] = A[0] * c - lb * cp - hb * cm  # step 4
    arr = np.argsort(-R[0])
    arr = arr[0]
    print(raw)
    for i in range(20):
        print(dic[arr[i]], end=" ")
    print()
if __name__ == '__main__':
    dataset_train=SPAM_Dataset(train=True)
    dataset_test = SPAM_Dataset(train=False)
    for i in range(0,12000):
        dic.append(0)
    for i in dataset_test.word2id:
        dic[dataset_test.word2id[i]]=i
    trainloader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=64, shuffle=True,
                                              num_workers=6)
    testloader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=64, shuffle=True,
                                              num_workers=6)
    explainloader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=64, shuffle=True,
                                              num_workers=6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net()
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=5e-3,weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.7)

    train=0     #do we need training

    if(train):
        for epoch in range(3):  # loop over the dataset multiple times
            train(net,device,trainloader,optimizer,criterion)
            test(net,device,testloader)
        decisiontree(dataset_train.rawdata, dataset_test.rawdata, dataset_train.labels, dataset_test.labels)

    for i in range(100, 500):
        data, target, raw = dataset_test[i]
        if (target.item()):
            break
    LRP('best.pkl',data,target)     #train your model and put it here

#DNN 99.13%
#decision tree 96.02%
