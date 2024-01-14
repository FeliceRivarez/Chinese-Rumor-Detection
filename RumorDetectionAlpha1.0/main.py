import pandas as pd
import jieba as ji
import csv
import math

# 一些变量的初始化
trainingSet = pd.read_csv('./train.news.csv')
words = dict()
statFake, statTrue, accountTrue, accountFake, contextTrue, contextFake = [], [], [], [], [], []
totalFake, totalTrue, totalWords, totalAccountTrue, totalAccountFake, totalAccounts, totalContexts = 0, 0, 0, 0, 0, 0, 0
account = dict()
context = []
stopwords = pd.read_csv('./stopwords.csv', quoting=csv.QUOTE_NONE)
stop = set()

# 朴素贝叶斯的训练
for i in range(len(stopwords)):
    stopTemp = str(stopwords.iloc[i].value)#停词表
    stop.add(stopTemp)

for i in range(len(trainingSet)):
    wordSet = ji.lcut_for_search(trainingSet.iloc[i].Title)#分词
    accountSet = trainingSet.iloc[i].account# 提取公众号名称，构建公众号词典
    result = trainingSet.iloc[i].label
    if result == 1:
        totalFake += 1
    else:
        totalTrue += 1
    i1 = 0
    for i2 in range(len(wordSet)):#构建编码词典
        if wordSet[i2] in stop:
            continue
        if wordSet[i2] not in words:
            words[wordSet[i2]] = totalWords
            totalWords += 1
            statTrue.append(0)
            statFake.append(0)
        if result == 0:
            statTrue[words[wordSet[i2]]] += 1 #统计频数
        if result == 1:
            statFake[words[wordSet[i2]]] += 1
    if accountSet not in account:
        account[accountSet] = totalAccounts #公众号词典构建
        totalAccounts += 1
        accountTrue.append(0)
        accountFake.append(0)
    if result == 0:
        totalAccountTrue += 1 #统计频数
        accountTrue[account[accountSet]] += 1
    if result == 1:
        totalAccountFake += 1
        accountFake[account[accountSet]] += 1

borderline = 0.3  # 标题概率门槛
vagueWord = set()
for i in words:
    currFake = float(statFake[words[i]]) / float(totalFake)
    currTrue = float(statTrue[words[i]]) / float(totalTrue)
    if currTrue == 0:
        currTrue = 0.1 / float(totalFake)
    if currFake == 0:
        currFake = 0.1 / float(totalTrue)
    if abs((currFake - currTrue) / (currFake + currTrue)) <= borderline:
        vagueWord.add(i)

contextBorder = 0.8

# 以下为生成半朴素贝叶斯的上下文，已经注释掉，因为已经保存了本地的上下文。
# 如果需要现场生成的话，取消注释即可

trainingSet = trainingSet.sample(frac=1)
for i in range(len(trainingSet)):
    if i % 100 == 0:
        print(i)
    wordSetVague = ji.lcut(trainingSet.iloc[i].Title)
    accountSet = trainingSet.iloc[i].account
    result = trainingSet.iloc[i].label
    if result == 1:
        totalFake += 1
    else:
        totalTrue += 1
    i1 = 0
    for i2 in range(len(wordSetVague)):#生成上下文词典
        if wordSetVague[i2] in vagueWord:#根据贝叶斯的概率，不显著的词也应该剔除
            continue
        if wordSetVague[i2] in stop:
            continue
        for i3 in range(len(wordSetVague)):
            if i3 <= i2 or wordSetVague[i3] in stop:#同样应用停词表
                continue
            if wordSetVague[i3] in vagueWord:
                continue
            if {wordSetVague[i2], wordSetVague[i3]} not in context:
                context.append({wordSetVague[i3], wordSetVague[i2]})
                contextTrue.append(0)
                contextFake.append(0)
                totalContexts += 1
            temp = context.index({wordSetVague[i3], wordSetVague[i2]})
            if result == 0:
                contextTrue[temp] += 1#统计频率
            if result == 1:
                contextFake[temp] += 1

delete = 0
print(len(context))
delList = []
inProgress = 0
for i in context:
    inProgress += 1
    # if inProgress%100==0:
    #     print(inProgress)
    temp = context.index(i)
    if contextTrue[temp] <= 1 and contextFake[temp] <= 1:
        delList.append(i)
        continue
    currTrue = float(contextTrue[temp]) / float(totalTrue)
    currFake = float(contextFake[temp]) / float(totalFake)
    if currTrue == 0:
        currTrue = 1 / float(totalFake)
    if currFake == 0:
        currFake = 1 / float(totalTrue)
    if contextTrue[temp] < contextFake[temp] and 2 * contextTrue[temp] + 2 > contextFake[temp]:
        delList.append(i)
        continue
    if contextTrue[temp] > contextFake[temp] and 2 * contextFake[temp] + 2 > contextTrue[temp]:
        delList.append(i)
        continue
inProgress = 0
for i in delList:
    inProgress += 1
    # if inProgress%100==0:
    #     print(inProgress)
    temp = context.index(i)
    del context[temp]
    del contextTrue[temp]
    del contextFake[temp]
    delete += 1
print(context)
print("deleted:" + str(delete))
print(len(context))
savedContext=open('./context.txt','w',encoding="utf-8")
contextData=open('./contextData.txt','w')
# 以上为生成上下文

#以下为保存上下文
for i in range(len(context)):
    for i1 in context[i]:
        savedContext.write(i1+"AND")
    savedContext.write('\n')
    contextData.write(str(contextTrue[i])+' ')
    contextData.write(str(contextFake[i]))
    contextData.write('\n')
#以上为保存上下文

# # 以下为加载本地上下文
# externalContext=open('./context.txt','r',encoding='utf-8')
# externalData=open('./contextData.txt','r')
# for i in externalContext:
#     tempSet= set(map(str,i.rstrip().split("AND")))
#     tempSet.remove('')
#     context.append(tempSet)
#     print(tempSet)
# for i in externalData:
#     tempTrue,tempFake=map(int,i.split())
#     contextTrue.append(tempTrue)
#     contextFake.append(tempFake)
# print(context)
# # 以上为加载本地上下文

testingSet = pd.read_csv('./test.feature.csv')
# testingSet = testingSet.sample(frac=1)
output = open('./result.txt', 'w')
output.write("id,label")
distribution=open('./prob.txt','w')# 输出概率
distribution.write("id,naiveTrue,naiveFake,halfTrue,halfFake,prob"+'\n')
output.write('\n')
accountBorder = 0.6  # 公众号概率门槛
none = 0  # 记录有多少组标题一个词都没法被计入概率，直接被跳过
skipped = 0  # 记录概率门槛筛掉了多少数据
trusted = 0  # 记录有多少公众号直接被信任
denied = 0  # 记录有多少公众号直接被拒绝
correctedFromTrue = 0  # 有多少拒绝改变了结果
correctedFromFake = 0  # 有多少信任改变了结果
labeledTrue = 0  # 总的预测结果
labeledFake = 0
for i in range(len(testingSet)):
    if i % 100 == 0:
        print("test" + str(i))
    probFake0 = 1.0  # 上下文得到的概率
    probTrue0 = 1.0
    probFake = 1.0  # 朴素贝叶斯得到的概率
    probTrue = 1.0
    wordSet = ji.lcut_for_search(testingSet.iloc[i].Title)
    wordSetVague = ji.lcut(testingSet.iloc[i].Title)
    accountSet = testingSet.iloc[i].account
    token_none = 0
    for i2 in range(len(wordSetVague)):
        if wordSetVague[i2] in vagueWord:
            continue
        for i3 in range(len(wordSetVague)):
            if wordSetVague[i3] in vagueWord:
                continue
            if i3 <= i2 or wordSetVague[i3] in stop:
                continue
            if {wordSetVague[i2], wordSetVague[i3]} in context:
                temp = context.index({wordSetVague[i3], wordSetVague[i2]})
                currTrue = float(contextTrue[temp]) / float(totalTrue)
                currFake = float(contextFake[temp]) / float(totalFake)
            else:
                continue
            if currTrue == 0:
                currTrue = 0.1 / float(totalFake)
            if currFake == 0:
                currFake = 0.1 / float(totalTrue)
            if abs(currFake - currTrue) / (currFake + currTrue) > contextBorder:
                probFake0 *= currFake
                probTrue0 *= currTrue
                # print({wordSetVague[i2], wordSetVague[i3]})
                test_flag = 1
    for i1 in wordSet:#对每个词进行词典查找
        if i1 in words:
            if i1 in vagueWord:
                continue
            currFake = float(statFake[words[i1]]) / float(totalFake)#单独计算概率
            currTrue = float(statTrue[words[i1]]) / float(totalTrue)
            if currTrue == 0:
                currTrue = 0.1 / float(totalFake)#Laplace平滑
            if currFake == 0:
                currFake = 0.1 / float(totalTrue)
            if currFake/currTrue>2 or currTrue/currFake>3:
                token_none = 1
                probFake *= currFake
                probTrue *= currTrue
            else:
                skipped += 1
    if token_none == 0:
        none += 1
    if accountSet in account:
        currFake = accountFake[account[accountSet]]
        currTrue = accountTrue[account[accountSet]]
        if currTrue == 0:
            currTrue = 0.1 / float(totalFake)
        if currFake == 0:
            currFake = 0.1 / float(totalTrue)
        probFake *= currFake
        probTrue *= currTrue
        # if float((currTrue - currFake) / (currFake + currTrue)) > 0.8:#相信公众号
        #     # print("trusted: " + str(accountSet))
        #     if probTrue < probFake:
        #         correctedFromFake += 1
        #     probTrue += 1
        #     trusted += 1
        if currFake/currTrue > 0.5:  # 不信任公众号
            # print("denied: " + str(accountSet))
            if probTrue > probFake:
                correctedFromTrue += 1
            probFake += 1
            denied += 1
    output.write(str(testingSet.iloc[i].id))
    # output.write(str(i))
    # print(str(i)+str(testingSet.iloc[i].Title))
    output.write(',')
    distribution.write(str(testingSet.iloc[i].id))
    # distribution.write(str(i))
    distribution.write(',')
    finalFake = probFake * probFake0  # 最终判定概率，现在已经废弃
    finalTrue = probTrue * probTrue0
    if probTrue != 0 and probTrue0!=0 and probFake!=0 and probFake0!=0:
        distribution.write(str(math.log(probFake)-math.log(probTrue)) +',' + str(math.log(probFake0)-math.log(probTrue0)))
        # distribution.write(str(math.log(probTrue)) + ',' + str(math.log(probFake)) + ',' + str(math.log(probTrue0)) + ',' + str(math.log(probFake0)) + ',')
        tempProb=math.log(probFake/probTrue,10)*0.3+math.log(probFake0/probTrue0,10)*0.7
        # distribution.write(str(tempProb))
        distribution.write('\n')
        # print(tempProb)
        if tempProb > -1:
            output.write("1")
            output.write('\n')
            labeledFake += 1
        else:
            output.write("0")
            output.write('\n')
            labeledTrue += 1
    else:
        if probTrue==0 or probTrue0==0:
            output.write("1")
            output.write('\n')
            distribution.write('100,100')
            distribution.write('\n')
            labeledFake += 1
        else:
            output.write("0")
            output.write('\n')
            distribution.write('100,-100,100,-100,-100')
            distribution.write('\n')
            labeledTrue += 1

print("undetermined:" + str(none))
print("fake:" + str(labeledFake))
print("true:" + str(labeledTrue))
print("skipped:" + str(skipped))
print("trusted:" + str(trusted))
print("denied:" + str(denied))
print("correctedFromFake:" + str(correctedFromFake))
print("correctedFromTrue:" + str(correctedFromTrue))

