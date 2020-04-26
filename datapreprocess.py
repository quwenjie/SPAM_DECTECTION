import os
path='.\data'
def readfile(file):
    with open(file, 'r') as f:
        str=f.read()
        str=str.lower()
        for i in range(len(str)):
            if not str[i].islower():
                str=str[:i]+' '+str[i+1:]
        return str.split(),str

def walkfile(file):
    test_dictionary=dict()
    train_dictionary=dict()
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        for f in files:
            filename=os.path.join(root, f)
            subwordvec,st=readfile(filename)

            istest=root[-1]=='1' or root[-1]=='2'

            for w in subwordvec:
                if(len(w)<=1):
                    continue
                if istest:
                    if not test_dictionary.get(w,0):
                        test_dictionary[w]=1
                    else:
                        test_dictionary[w]+=1
                else:
                    if not train_dictionary.get(w, 0):
                        train_dictionary[w]=1
                    else:
                        train_dictionary[w]+=1
    cnt=0
    return test_dictionary,train_dictionary

def calculate_wordid():
    test,train=walkfile(path)
    word2id=dict()
    cnt=1
    for i in test:
        if train.get(i,0):
           word2id[i]=cnt
           cnt+=1
    return word2id

