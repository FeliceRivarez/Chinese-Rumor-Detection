import sklearn as sk
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from mlxtend.classifier import StackingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import jieba as ji
import csv
import torch
from sklearn.linear_model import LinearRegression as LR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import math
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from sklearn.svm import SVC

trainingSet = pd.read_csv('./train.news.csv')
# trainingSet=trainingSet.sample(frac=0.1)
words = dict()
statFake, statTrue, accountTrue, accountFake, contextTrue, contextFake = [], [], [], [], [], []
totalFake, totalTrue, totalWords, totalAccountTrue, totalAccountFake, totalAccounts, totalContexts = 0, 0, 0, 0, 0, 0, 0
account = dict()
context = []
stopwords = pd.read_csv('./stopwords.csv', quoting=csv.QUOTE_NONE)
stop = set()
# clf=sk.svm.SVC(kernel="linear",C=1.0,gamma="auto")
# clf.fit([1,0],[1])
# 11.9提示：强筛上下文，优中选优
for i in range(len(stopwords)):
    stopTemp = str(stopwords.iloc[i].value)
    stop.add(stopTemp)

for i in range(len(trainingSet)):
    wordSet = ji.lcut_for_search(trainingSet.iloc[i].Title)
    accountSet = trainingSet.iloc[i].account
    result = trainingSet.iloc[i].label
    if result == 1:
        totalFake += 1
    else:
        totalTrue += 1
    i1 = 0
    for i2 in range(len(wordSet)):
        if wordSet[i2] in stop:
            continue
        if wordSet[i2] not in words:
            words[wordSet[i2]] = totalWords
            totalWords += 1
            statTrue.append(0)
            statFake.append(0)
        if result == 0:
            statTrue[words[wordSet[i2]]] += 1
        if result == 1:
            statFake[words[wordSet[i2]]] += 1
    if accountSet not in account:
        account[accountSet] = totalAccounts
        totalAccounts += 1
        accountTrue.append(0)
        accountFake.append(0)
    if result == 0:
        totalAccountTrue += 1
        accountTrue[account[accountSet]] += 1
    if result == 1:
        totalAccountFake += 1
        accountFake[account[accountSet]] += 1

initArray=list(range(len(words)))
for i in initArray:
    initArray[i]=0

# 以下为生成训练集向量，由于可以加载已经处理过的训练集向量，所以已经注掉
# ndarray=np.array(initArray)
# temp=0
# for i in range(len(trainingSet)):
#     temp+=1
#     if temp%100==0:
#         print(temp)
#     newArray=np.array(initArray)
#     wordSet = ji.lcut_for_search(trainingSet.iloc[i].Title)
#     for i1 in wordSet:
#         if i1 in words:
#             newArray[words[i1]]=1
#     ndarray=np.vstack((ndarray,newArray))
# ndarray = np.delete(ndarray, 0, axis=0)
# 保存已经生成的训练集向量
# np.savetxt('./processedTrained.txt',ndarray,fmt="%d",delimiter=',')

# 加载已经处理过的训练集向量
ndarray=np.loadtxt('./processedTrained.txt',dtype=int,delimiter=',')
print(len(ndarray[0]))

# 生成测试集向量
testingSet = pd.read_csv('./test.feature.csv')
# ndarray2=np.array(initArray)
# print(len(initArray))
# temp=0
# for i in range(len(testingSet)):
#     temp+=1
#     if temp%100==0:
#         print(temp)
#     wordSet = ji.lcut_for_search(testingSet.iloc[i].Title)
#     newArray = np.array(initArray)
#     for i1 in wordSet:
#         if i1 in words:
#             newArray[words[i1]] = 1
#     ndarray2 = np.vstack((ndarray2, newArray))
# ndarray2 = np.delete(ndarray2, 0, axis=0)
# 保存测试集向量
# np.savetxt('./processedTest.txt',ndarray2,fmt="%d",delimiter=',')

# 加载已经保存的测试集向量
ndarray2=np.loadtxt('./processedTest.txt',dtype=int,delimiter=',')
print("done")

# 对不平衡的数据集进行random over sample
res = RandomOverSampler(random_state = 1, sampling_strategy='auto')

ndarray,label=res.fit_resample(ndarray,trainingSet.label)
# ndarray=PCA(n_components=5000).fit_transform(ndarray)
# ndarray2=PCA(n_components=5000).fit_transform(ndarray2)

#下面的模型，需要哪个就对哪个取消注释
#随机森林
rf=RandomForestClassifier(n_estimators=50, random_state=0)
rf.fit(ndarray, label)
joblib.dump(rf, "rf.pkl")
rf=joblib.load("rf.pkl")
print("model loaded")

#KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(ndarray, trainingSet.label)
joblib.dump(knn, "knn.pkl")
# knn=joblib.load("knn.pkl")
# print("model loaded")

#BP神经网络
mlp = MLPClassifier(hidden_layer_sizes=(10,100,100,100,10), max_iter=200, random_state=0,verbose=100)
mlp.fit(ndarray, label)
joblib.dump(mlp, "mlp.pkl")
# mlp=joblib.load("mlp.pkl")
# print("model loaded")

# gb=GaussianNB()
# gb.fit(ndarray, label)

# staking融合模型
# rf=RandomForestClassifier(n_estimators=10, random_state=0)
# # knn = KNeighborsClassifier(n_neighbors = 15)
# mlp = MLPClassifier(hidden_layer_sizes=(10,100,100,100,10), max_iter=1000, random_state=0,verbose=100)
# # clf=SVC(C=1.0,kernel='linear',gamma="auto")

sclf = StackingClassifier(classifiers=[mlp,rf], meta_classifier=rf,verbose=10)
print("done")
joblib.dump(sclf, "sclf.pkl")
#
# sclf.fit(ndarray, label)

# lr=LR()
# lr.fit(ndarray, label)
# joblib.dump(lr,"lr.pkl")

result=open('./result.csv','w')
result.write("id,label"+'\n')
temp=0
ID=0
true=0
fake=0
for i in ndarray2:
    temp+=1
    if temp%1==0:
        print(temp)
        result.close()
        result=open('./result.csv','a')
    tempArray=i.reshape(1,-1)
    prediction=(rf.predict(tempArray))
    ID+=1
    result.write(str(ID))
    result.write(',')
    if prediction[0]==0:
        result.write('0')
        result.write('\n')
        true+=1
    else:
        result.write('1')
        result.write('\n')
        fake+=1
print(true)
print(fake)