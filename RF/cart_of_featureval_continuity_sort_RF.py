# -*- coding:utf-8 -*-
"""
create on 2019-10

@author: qiuzefeng
"""
import csv
from numpy import *
from random import randrange
import operator

from show_matplotlip import creatPlot as cr


def readDataSet(filename):
    """
    读取数据
    :param filename:
    :return:
    """
    dataSet = []
    labels = []

    with open(filename) as file:
        curLien = file.readline()
        line1 = curLien.strip("\n").split("	")
        labels.extend(line1[: -1])
        for line in file.readlines():
            fields = line.strip("\n").split("	")
            t = map(float, fields[2: -1])
            t = list(map(float,t))
            t.append(fields[-1])
            dataSet.append(t)

    return dataSet, labels

def ReadDataSet(csvfilename):
    """
    读取csv文件
    :param csvfilename: 文件名或路径
    :return: 数据及特征
    """
    dataSet = []
    labels = []

    with open(csvfilename) as csvfile:
        # 读取csvfile中的文件
        csv_reader = csv.reader(csvfile)
        # 读取第一行每一列的标题
        label = next(csv_reader)
        labels = label[: -1]
        for row in csv_reader:
            ro = map(float, row[: -1])
            ro.append(row[-1])
            dataSet.append(ro)

    return dataSet, labels

def getTrainDataAndPredictData(dataSets):
    """
    把数据按7：3分成训练集与测试集
    :param dataSet:
    :return:
    """
    length = len(dataSets)

    # 产生长度为length的序列
    array = arange(length)
    # 将序列随机排列
    random.shuffle(array)
    # array = selIndex

    predictData = []
    trainData = []

    for index in array:
        if len(trainData) <= length * 0.7:
            trainData.append(dataSets[index])
        else:
            predictData.append(dataSets[index])

    return trainData, predictData


def calGini(dataSet):
    """
    计算Gini指数
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    gini = 0
    for label in labelCounts.keys():
        prop = float(labelCounts[label]) / numEntries
        gini -= prop * prop

    return gini

def splitDataSet(dataSet, axis, value, threshold):
    """
    划分数据
    :param dataSet:数据集
    :param axis:特征序号
    :param values:二分的特征值列表
    :return:带二分特征值的数据集
    """
    retDataSet = []
    if threshold == "lt":
        for featVec in dataSet:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[: axis]
                reducedFeatVec.extend(featVec[axis+1: ])
                retDataSet.append(reducedFeatVec)
    else:
        for featVec in dataSet:
            if featVec[axis] > value:
                reducedFeatVec = featVec[: axis]
                reducedFeatVec.extend(featVec[axis+1: ])
                retDataSet.append(reducedFeatVec)

    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最佳特征
    :param dataSet: 数据集
    :param m: 随机选择特征的个数m
    :return: 最佳特征序号，最佳特征的二分特征值，最好的gini系数
    """
    index = []
    numFeatures = len(dataSet[0]) - 1
    bestGini = 1.0
    bestFeature = -1
    bestBinaryFeatVal = ""

    # 在numFeatures个特征里随机选择m个特征，然后在这m个特征里选择一个合适的分类特征。
    m = int(math.log(numFeatures, 2))
    index = random.choice(a=range(numFeatures), size=m, replace=False, p=None)

    for i in index:
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList))
        uniqueVals.sort()
        uniqueVals = uniqueVals[: -1]
        for value in uniqueVals:
            GiniGain = 0.0
            #左增益
            left_subDataSet = splitDataSet(dataSet, i, value, 'lt')
            left_prob = len(left_subDataSet) / float(len(dataSet))
            GiniGain += left_prob * calGini(left_subDataSet)
            #右增益
            right_subDataSet = splitDataSet(dataSet, i, value, 'gt')
            right_prob = len(right_subDataSet) / float(len(dataSet))
            GiniGain += right_prob * calGini(right_subDataSet)

            if (GiniGain <= bestGini):
                # 记录最好的结果和做好的特征
                bestGini = GiniGain
                bestFeature = i
                bestBinaryFeatVal = value

    return bestFeature, bestBinaryFeatVal, bestGini

def majorityCnt(classList):
    """
    多数表决
    :param classList:类别列表
    :return:表决结果
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0]

def creatTree(dataSet, labels, max_level = 10):
    """
    构建CART树
    :param dataSet: 数据集
    :param labels: 特征
    :param m: 随机选择的特征个数
    :param max_level: 树的深度
    :return: CART树
    """
    classList = [example[-1] for example in dataSet]

    # 如果所有类别都一样，建立节点
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 如果选择最佳的特征比不选的好则不选，直接建立节点
    if len(dataSet) == 1:
        return majorityCnt(classList)

    bestFeat, bestBinarySplitVal, bestGini = chooseBestFeatureToSplit(dataSet)

    # 如果所有特征Gini都比没划分时的阈值好，多数表决决定类别
    if bestFeat == -1:
        return majorityCnt(classList)

    bestFeatLabel = labels[bestFeat] + ':' + str(bestBinarySplitVal)

    myTree = {bestFeatLabel: {}}

    max_level -= 1
    # 如果深度小于设定深度，多数表决
    if max_level < 0:
        return majorityCnt(classList)

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    subLabels = labels[:]  # 拷贝防止labels被修改
    del subLabels[bestFeat]
    myTree[bestFeatLabel]['<=' + str(round(float(bestBinarySplitVal), 3))] = creatTree(
        splitDataSet(dataSet, bestFeat, bestBinarySplitVal, 'lt'), subLabels, max_level)
    myTree[bestFeatLabel]['>' + str(round(float(bestBinarySplitVal), 3))] = creatTree(
        splitDataSet(dataSet, bestFeat, bestBinarySplitVal, 'gt'), subLabels, max_level)

    return myTree

def labelsDict(labels):
    """
    把特征与索引建立字典
    :param labels:
    :return:
    """
    length = len(labels)
    labelsOfDict = {}

    for i in range(length):
        currentLabel = labels[i]
        if currentLabel not in labelsOfDict.keys():
            labelsOfDict[currentLabel] = i

    return labelsOfDict

def subsample(dataset):
    """
    样本有放回随机采样,输入数据集，返回采样子集'
    :param dataset: 数据集
    :return:随机抽样的后的数据集
    """
    sample=list()
    indexs = []
    n_sample=len(dataset)
    while len(sample)<n_sample:
        index=randrange(len(dataset))
        indexs.append(index)
        sample.append(dataset[index])

    return sample

def RondomForest(dataSet, labels, n, max_level):
    """
    随机森林
    :param dataSet: 数据集
    :param labels: 特征
    :param n: 决策树棵数（n个子集）
    :param max_level: 树的深度
    :return: 森林
    """
    Trees = []
    for i in range(n):
        dataSet = subsample(dataSet)
        tree = creatTree(dataSet, labels, max_level)
        Trees.append(tree)

    return Trees

def predict(tree, predictData, labelsOfDict):
    """
    预测
    :param tree: 决策树
    :param sample: 预测数据
    :param labelsOfDict: 特征字典索引
    :return: 预测值
    """
    if type(tree) is not dict:
        return tree
    root = tree.keys()[0]
    # 取出根节点，得到最优特征和阀值
    feature, threshold = root.split(':')
    threshold = float(threshold)
    # 递归预测
    if predictData[labelsOfDict[feature]] > threshold:
        return predict(tree[root]['>' + str(round(float(threshold), 3))], predictData, labelsOfDict)
    else:
        return predict(tree[root]['<=' + str(round(float(threshold), 3))], predictData, labelsOfDict)

def getPredictY(trees, predictDatas, labelsOfDict):
    """
    得到所有树的预测值
    :param trees: 森林
    :param samples: 测试集
    :param predictLabelsDict: 特征索引字典
    :return: 每棵树的Y
    """
    predictY = []

    length = len(trees)
    for samplerow in predictDatas:
        predictAlltreeY = []
        for i in range(length):
            Y = predict(trees[i], samplerow, labelsOfDict)
            predictAlltreeY.append(Y)
        predictAlltreeMaxY = majorityCnt(predictAlltreeY)
        predictY.append(predictAlltreeMaxY)

    return predictY

def correctRate(y, predictY):
    """
    计算预测的准确率
    :param y: 预测集y
    :param predictY: 预测结果
    :return: 预测准确率
    """
    length = len(y)
    count = 0

    for index in range(length):
        if y[index] == predictY[index]:
            count += 1

    corr = count / float(length)

    return corr


def result(path, tree_num, tree_depth):

    dataSet, labels = ReadDataSet(path)

    # 得到训练集与测试集
    trainData, predictData = getTrainDataAndPredictData(dataSet)

    # 得到预测集的类别
    y = [example[-1] for example in predictData]

    # 产生森林 参数（训练集，特征，方差，树的个数，随机选择特征的个数，树的森度）
    Trees = RondomForest(trainData, labels, tree_num, tree_depth)

    # 预测 先把特征与索引建立字典为预测时找到对应的数据
    labelsOfDict = labelsDict(labels)

    # 得到每棵树的应变量的值 参数(森林，测试集，特征索引字典)
    predictY = getPredictY(Trees, predictData, labelsOfDict)

    # 得到准确率
    corr = correctRate(y, predictY)

    return corr

if __name__ == '__main__':
    path = "../datas/splice.csv"
    corr = result(path, 20, 15)
    print corr