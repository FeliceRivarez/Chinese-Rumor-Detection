import pandas as pd
import jieba as ji
import csv


stopwords=pd.read_csv('./stopwords.csv',quoting=csv.QUOTE_NONE)
stop=set()
for i in range(len(stopwords)):
    stopTemp = str(stopwords.iloc[i].value)
    stop.add(stopTemp)
# testing=pd.read_csv('./test.feature.csv')
training=pd.read_csv('./test.csv')

parsedTrain=open('./test.feature.csv','w',encoding="utf-8")
parsedTrain.write("id,Title"+'\n')
# parsedTest=open('./parsedTest.csv','w',encoding="utf-8")
# parsedTest.write("id,Title"+'\n')


# print(stopwords.value)

for i in range(len(training)):
    curr=training.iloc[i]
    tempStr=training.iloc[i].content
    tempStr=tempStr.replace(',',"，")
    words=ji.lcut_for_search(tempStr)
    newWords=list()
    for i1 in words:
        if i1 not in stop:
            newWords.append(i1)
    parsedTrain.write(str(training.iloc[i].id)+',')
    for i1 in newWords:
        parsedTrain.write(i1+" ")
    # parsedTrain.write(','+str(training.iloc[i].label)+'\n')
    parsedTrain.write('\n')

parsed=pd.read_csv('./test.feature.csv')
for i in range(len(parsed)):
    if type(parsed.iloc[i].Title)!=str:
    # if type(training.iloc[i].content)!=str or type(training.iloc[i].label)!=str or type(training.iloc[i].id)!=str:
        print(i)


# for i in range(len(testing)):
#     words=ji.lcut_for_search(testing.iloc[i].content)
#     newWords=list()
#     for i1 in words:
#         if i1 not in stop:
#             newWords.append(i1)
#     parsedTest.write(str(testing.iloc[i].id)+',')
#     for i1 in newWords:
#         parsedTest.write(i1+" ")
#     parsedTest.write('\n')