import pandas as pd
import numpy as np
import joblib
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn as sk
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import imblearn
from mlxtend.classifier import StackingClassifier, LogisticRegression

trainingSet=pd.read_csv('./trainingSetProb.csv')
testingSet=pd.read_csv('./testingSetProb.csv')
answer=pd.read_csv('./trainingSetKey.csv')

# 概率二维向量的生成，由于本地已经保存所以此处注掉
# train=open('trainingArray.txt','w')
# test=open('testingArray.txt','w')
# for i in range(len(trainingSet)):
#     train.write(str(trainingSet.iloc[i].naiveFake-trainingSet.iloc[i].naiveTrue))
#     train.write(',')
#     train.write(str(trainingSet.iloc[i].halfFake-trainingSet.iloc[i].halfTrue))
#     train.write('\n')
# for i in range(len(testingSet)):
#     test.write(str(testingSet.iloc[i].naiveFake - testingSet.iloc[i].naiveTrue))
#     test.write(',')
#     test.write(str(testingSet.iloc[i].halfFake - testingSet.iloc[i].halfTrue))
#     test.write('\n')

ndarray=np.loadtxt('./trainingArray.txt',dtype=float,delimiter=',')
ndarray2=np.loadtxt('./testingArray.txt',dtype=float,delimiter=',')

# 不平衡数据集的调整
res = BorderlineSMOTE(sampling_strategy='auto', random_state = 0, k_neighbors = 4)
# res = ADASYN(sampling_strategy='auto',random_state=0,n_neighbors=5)
# res = SMOTE(sampling_strategy='auto', random_state = 1, k_neighbors = 3)

ndarray,label=res.fit_resample(ndarray,answer.label)

print(len(ndarray))

#以下模型中，需要使用哪个，把注释取消掉即可

# 以下为支持向量机
# clf=svm.SVC(kernel="linear",C=1,gamma="auto",probability=False)
# clf.fit(ndarray,label)
# joblib.dump(clf,"svmIntegration.pkl")
# clf=joblib.load("svmIntegration.pkl")
# print("model loaded")


# 以下为随机森林
# rf=RandomForestClassifier(n_estimators=100, random_state=0)
# rf.fit(ndarray, label)
# joblib.dump(rf, "rf.pkl")
# rf=joblib.load("rf.pkl")
# print("model loaded")

# 以下为KNN
# knn = KNeighborsClassifier(n_neighbors = 3)
# knn.fit(ndarray, label)
# joblib.dump(knn, "knn.pkl")
# knn=joblib.load("knn.pkl")
# print("model loaded")

# 以下为BP神经网络
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=0,verbose=100)
mlp.fit(ndarray, label)
joblib.dump(mlp, "mlp.pkl")
mlp=joblib.load("mlp.pkl")
print("model loaded")

# 以下为stacking融合模型，可以根据需要调整融合的模型数目
# rf=RandomForestClassifier(n_estimators=100, random_state=0)
# knn = KNeighborsClassifier(n_neighbors = 15)
# mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=0,verbose=100)
#
# sclf = StackingClassifier(classifiers=[knn, mlp], meta_classifier=rf,verbose=100)
# print("done")
# sclf.fit(ndarray, label)

# 以下为结果输出
result=open('./result.csv','w')
result.write("id,label"+'\n')
temp=0
ID=0
true=0
fake=0
for i in ndarray2:
    temp+=1
    if temp%100==0:
        print(temp)
        result.close()
        result=open('./result.csv','a')
    tempArray=i.reshape(1,-1)
    prediction=knn.predict(tempArray)
    # prob=clf.predict_proba(tempArray)
    # print(prob)
    result.write(str(int(testingSet.iloc[ID].id)))
    ID += 1
    # result.write(str(ID))
    result.write(',')
    if prediction==0:
        result.write('0')
        # result.write(','+str(prob))
        result.write('\n')
        true+=1
    else:
        result.write('1')
        # result.write(','+str(prob))
        result.write('\n')
        fake+=1
print(fake)
print(true)


# 废用代码
# result=pd.read_csv('./result.csv')
# output=open("./result2.csv",'w')
# output.write("id,label"+'\n')
# for i in range(len(result)):
#     if ndarray2[i,0]<-10 and ndarray2[i,1]<-2:
#         output.write(str(int(result.iloc[i].id))+','+'0'+'\n')
#     elif ndarray2[i,0]>2 and ndarray2[i,1]>2:
#         output.write(str(int(result.iloc[i].id)) + ',' + '1'+'\n')
#     else:
#         output.write(str(int(result.iloc[i].id))+','+str(result.iloc[i].label)+'\n')


