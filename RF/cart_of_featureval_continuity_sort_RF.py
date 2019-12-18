# -*- coding:utf-8 -*-
"""
create on 2019-10

@author: qiuzefeng
"""
import csv
import operator
from numpy import  *
from random import randrange

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
        #读取csvfile中的文件
        csv_reader = csv.reader(csvfile)
        # 读取第一行每一列的标题
        label = next(csv_reader)
        labels = label[: -1]
        for row in csv_reader:
            ro = map(float, row[: -1])
            ro.append(row[-1])
            dataSet.append(ro)

    return dataSet, labels


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
                #记录最好的结果和做好的特征
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

def creatTree(dataSet, labels, m = 20, max_level = 10):
    """
    构建CART树
    :param dataSet: 数据集
    :param labels: 特征
    :param m: 随机选择的特征个数
    :param max_level: 树的深度
    :return: CART树
    """
    classList = [example[-1] for example in dataSet]

    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet) == 1:
        majorityCnt(classList)

    bestFeat, bestBinarySplitVal, bestGini = chooseBestFeatureToSplit(dataSet)
    #如果所有特征Gini都一样，多数表决决定类别
    if bestFeat == -1:
        return majorityCnt(classList)

    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}

    max_level -= 1
    #如果深度小于设定深度，多数表决
    if max_level < 0:
        return majorityCnt(classList)

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    subLabels = labels[:]  # 拷贝防止labels被修改
    del subLabels[bestFeat]
    myTree[bestFeatLabel][bestFeatLabel + '<=' + str(round(float(bestBinarySplitVal), 3))] = creatTree(
        splitDataSet(dataSet, bestFeat, bestBinarySplitVal, 'lt'), subLabels, m, max_level)
    myTree[bestFeatLabel][bestFeatLabel + '>' + str(round(float(bestBinarySplitVal), 3))] = creatTree(
        splitDataSet(dataSet, bestFeat, bestBinarySplitVal, 'gt'), subLabels, m, max_level)

    return myTree

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

def main():
    path = "../datas/splice.csv"
    dataSet, labels = ReadDataSet(path)
    print "{0}\n{1}".format(dataSet, labels)

    Trees = RondomForest(dataSet, labels, 10, 6)
    #输出每一颗决策树
    # for i in range(10):
    #     print Trees[i]
        # cr.createPlot(Trees[i])


    # myTree = creatTree(dataSet, labels, 5, 4)
    # print myTree
    # cr.createPlot(myTree)


if __name__ == '__main__':
    main()
