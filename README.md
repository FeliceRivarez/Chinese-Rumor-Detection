NKU的同学们如果参考，别忘了点个Star~

# Chinese-Rumor-Detection

# 中文介绍
南开大学 计算机与网络空间安全学院 python语言程序设计大作业

## 任务说明：
1.数据集是中文微信公众号消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在train.news.csv，测试数据保存在test.feature.csv。

数据来源：[Wang, Y., Yang, W., Ma, F., Xu, J., Zhong, B., Deng, Q., & Gao, J. (2020). Weak Supervision for Fake News Detection via Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 34(01), 516-523.](https://github.com/yaqingwang/WeFEND-AAAI20)

2.需要根据训练集，来对测试集的新闻的真伪进行预测。

3.系统采用AUC进行评价

## 项目说明:
1.本项目采用了多种途径实现了大作业的任务。

  RumorDetectionAlpha1.0:

    手写的朴素贝叶斯与半朴素贝叶斯综合模型。可以进行调整的参数远大于sklearn等库的已有模型。
    
    预处理模式：onehot编码。
    
  RumorDetectionAlpha2.0:

    包括sklearn实现的随机森林、支持向量机、BP神经网络分类模型。

    预处理方式：onehot编码
    
  RumorDetectionAlpha3.0:

    包括sklearn实现的KNN、随机森林、支持向量机、BP神经网络分类模型。

    预处理方式：利用RumorDetectionAlpha1.0对每一条数据产生二维条件概率
    
  RumorDetectionAlpha4.0:

    包括Keras库实现的CNN、RNN、CNN-RNN、LSTM、BiLSTM分类模型。

    预处理方式：tokenizing+wordEmbedding

  RumorDetectionBeta5.0:

    包括Hugging Face的Transformers库实现的BERT分类模型，预训练模型使用了bert-base-chinese

    预处理方式：tokenizing+wordEmbedding


