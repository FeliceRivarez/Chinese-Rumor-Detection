# import sklearn as sk
# from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
# from mlxtend.classifier import StackingClassifier
# from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# import pandas as pd
# import jieba as ji
# import csv
# import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LinearRegression as LR
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# import math
# import joblib
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import PCA
# from imblearn.over_sampling import BorderlineSMOTE, SMOTE
# from sklearn.svm import SVC
#
# train=pd.read_csv('./parsedTrain.csv')
# train=train.sample(frac=1)
# docs=train['Title'].fillna("")
# targets=np.asarray(train["label"])
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(docs)
# print(X)
#
# res = RandomOverSampler(random_state = 1, sampling_strategy='auto')
# X,targets=res.fit_resample(X,targets)
#
#
# # labeled=pd.read_csv('./parsedLabeled.csv')
# # docs1=labeled['Title'].fillna("")
# # print(docs)
# # answer=np.asarray(labeled["label"])
# # verify=pd.read_csv("./labeledVerification.csv")
# # docs_veri=verify["Title"].fillna("")
# testing=pd.read_csv("./parsedTest.csv")
# docs_test=testing["Title"].fillna("")
# # verifyAnswer=np.asarray[verify["label"]]
#
# X_test=vectorizer.transform(docs_test)
#
#
#
# # clf=SVC(C=1.0,kernel='linear',gamma='auto')
# # clf.fit(X,targets)
#
# # rf=RandomForestClassifier(n_estimators=50, random_state=0)
# # rf.fit(X, targets)
#
# # mlp = MLPClassifier(hidden_layer_sizes=(10,100,100,10), max_iter=100, random_state=0,verbose=100,validation_fraction=0.1,early_stopping=True)
# # mlp.fit(X, targets)
#
# # lr=LR()
# # lr.fit(X, targets)
#
# result=open('./result.csv','w')
# result.write("id,label"+'\n')
# temp=0
# ID=0
# true=0
# fake=0
# for i in X_test:
#     temp+=1
#     if temp%1==0:
#         print(temp)
#         result.close()
#         result=open('./result.csv','a')
#     tempArray=i.reshape(1,-1)
#     prediction=(lr.predict(tempArray))
#     ID+=1
#     result.write(str(ID))
#     result.write(',')
#     if prediction[0]==0:
#         result.write('0')
#         result.write('\n')
#         true+=1
#     else:
#         result.write('1')
#         result.write('\n')
#         fake+=1
# print(true)
# print(fake)
from datasets import load_dataset
from sklearn.metrics._scorer import metric
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, get_scheduler
import pandas as pd
import numpy as np
import torch
import keras
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
import datasets as Datasets

print(torch.cuda.is_available())
device=torch.device('cuda:0')

print("1")
model_name = 'bert-base-chinese'
print("1")
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
print("1")
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=1)
print("1")
train=pd.read_csv('./train.news.csv')
# train=train.sample(frac=1)
temp=train['Title'].fillna('')
docs=temp.tolist()
# print(docs)
targets=np.asarray(train["label"])
# model.save_pretrained("model_path")

train['Title']=train['Title'].fillna("")
train['label']=train['label'].astype(float)
train.to_csv('./tempTrain.csv',index_label=False,index=False)

test=pd.read_csv('./verification.csv')
test['Title']=test['Title'].fillna("")
test['label']=test['label'].astype(float)
test.to_csv('./tempTest.csv',index_label=False,index=False)

data_files = {"train": './tempTrain.csv'}
ds=load_dataset("csv", data_files=data_files, delimiter=",")
data_files = {"test": './tempTest.csv'}
dsl=load_dataset('csv',data_files=data_files,delimiter=',')
# print(ds)

l=list()
for i in range(len(docs)):
    l.append(docs[i])
# print(l)
# test1=tokenizer(l,padding='max_length',max_length=60)
# print(test1['token_type_ids'])


def tokenize_function(examples):
  temp=examples['Title']
  for i in range(len(temp)):
    if type(temp[i])!=str:
      temp[i]='!'
      print(i)
  return tokenizer(temp, padding="max_length", truncation=True, max_length=60)


tokenized_datasets = ds.map(tokenize_function, batched=True)
tokenized_datasets2 = dsl.map(tokenize_function, batched=True)

tokenized_datasets.remove_columns("Title")
tokenized_datasets2.remove_columns("Title")

print(tokenized_datasets)

tokenized_datasets.set_format("torch")
tokenized_datasets2.set_format("torch")



args = TrainingArguments(
  output_dir='validateSave',
  metric_for_best_model="accuracy",
  num_train_epochs=3
)

# print(tokenized_datasets)



def compute_metrics(eval_pred):
  logits, labels = eval_pred

  predictions = np.argmax(logits, axis=-1)

  return metric.compute(predictions=predictions, references=labels)

model.to(device)

trainer = Trainer(
  model,
  args=args,
  train_dataset=tokenized_datasets['train'],
  eval_dataset=tokenized_datasets2['test'],
  tokenizer=tokenizer,
  compute_metrics=compute_metrics
)

trainer.train()

model_path = "./finalSave/checkpoint-3000"
# model_path='./validateSave/checkpoint-500'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1).to("cuda")

def predict(input_text):
    # 对文本进行标记和编码
    if type(input_text)!=str:
      input_text="!"
    input_ids = tokenizer(input_text , padding="max_length", truncation=True, max_length=60, return_tensors='pt')
    input_ids.to("cuda")
    outputs = model(**input_ids)
    print(outputs)
    if(outputs.logits[0,0]>0.5):#!!!CHANGE HERE
      return(1)
    else:
      return(0)

result=open('./result.csv','w')
result.write("id,label"+'\n')
temp=0
ID=0
true=0
fake=0
parsedTrain=pd.read_csv("./test.feature.csv")
for i in range(len(parsedTrain)):
    temp+=1
    if temp%1==0:
        print(temp)
        result.close()
        result=open('./result.csv','a')
    ID+=1
    result.write(str(ID))
    result.write(',')
    prob=predict(parsedTrain.iloc[i].Title)
    print(prob)
    if prob==0:
      result.write('0')
      true+=1
    else:
      result.write('1')
      fake+=1
    result.write('\n')
print(true)
print(fake)
print(temp)

# encoding = tokenizer.encode_plus(
#             docs,
#             padding='max_length',
#             truncation=True,
#             add_special_tokens=True,
#             max_length=60,
#             return_token_type_ids=False,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#
# # docs=tokenizer(docs)
# print(encoding)