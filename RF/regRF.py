# -*- coding:utf-8 -*-

"""
create on: 2019-11

@author: qiuzefeng
"""


from random import randrange

from numpy import *


class RF():
    """
    随机森林
    """
    def __init__(self, treeNum, treeDepth, trainData, predictData, labels):
        """
        设置随机森林参数
        :param treeNum: 决策树的棵数
        :param treeDepth: 决策树深度
        :param trainData: 训练集
        :param predictData: 测试集
        :param labels:特征
        """
        self.treeNum = treeNum
        self.treeDepth = treeDepth
        self.trainData = trainData
        self.predictData = predictData
        self.labels = labels

    def binsplitDataSet(self, dataSet, feature, value):
        """
        切分数据集
        :param dataSet: 数据集
        :param feature: 最佳特征的序号
        :param value: 最佳二分特征值
        :return: mat0, mat1
        """
        mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0],:]
        mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:]

        return mat0, mat1

    def regErr(self, dataSet):
        """
        计算方差
        :param dataSet: 数据集
        :return: 方差
        """
        if len(dataSet) == 0:
            return 0
        return var(dataSet[:, -1])

    def regLeaf(self, dataSet):
        """
        计算平均值
        :param dataSet: 数据集
        :return: 平均值
        """
        return mean(dataSet[:, -1])

    def chooseBestFeature(self, dataSet):
        """
        选择最佳特征
        :param dataSet: 数据集
        :return: bestFeature（最佳特征索引） bestBinarySplitVal（最佳二分特征值）, bestStDev（二分后的最好方差）
        """
        numFeatures = dataSet.shape[1] - 1
        index = []
        bestStDev = inf
        bestFeature = -1
        bestBinarySplitVal = ""

        S = self.regErr(dataSet)

        # 在numFeatures个特征里随机选择m个特征，然后在这m个特征里选择一个合适的分类特征。
        m = int(math.log(numFeatures, 2))
        index = random.choice(a=range(numFeatures), size=m, replace=False, p=None)

        for feature in index:
            #得到每一个特征的特征值
            for splitVal in set(dataSet[:, feature]):
                #根据特征值切分数据
                mat0, mat1 = self.binsplitDataSet(dataSet, feature, splitVal)
                #切分数据后的总方差（左右方差和）
                newS = self.regErr(mat0) + self.regErr(mat1)

                if bestStDev > newS:
                    bestFeature = feature
                    bestBinarySplitVal = splitVal
                    bestStDev = newS
        #如果对所有特征划分后方差和没划分时相差不大则不选特征直接取平均值
        if (S - bestStDev) < 0.001:
            return None, self.regLeaf(dataSet), S

        return bestFeature, bestBinarySplitVal, bestStDev

    def createTree(self, dataSet, labels, max_level):
        """
        建树
        :param dataSet: 数据集
        :param max_level: 最大深度
        :return:
        """
        classList = set(dataSet[:, -1])

        # 如果所有Y都一样，停止划分
        if len(classList) == 1:
            return list(classList)[0]
        # 如果没有继续可以划分的特征，取平均值
        if dataSet.shape[0] == 1:
            return self.regLeaf(dataSet)
        # 如果深度小于设定深度，取平均值
        if max_level < 0:
            return self.regLeaf(dataSet)

        bestFeature, bestBinarySplitVal, bestStDev = self.chooseBestFeature(dataSet)

        # 如果对所有特征划分后方差和没划分时相差不大则不选特征直接取平均值作为叶子节点
        if bestFeature == None:
            return bestBinarySplitVal

        # 如果所有特征的方差都比没划分时，取平均值建立节点
        if bestFeature == -1:
            return self.regLeaf(dataSet)

        bestFeatLabel = labels[bestFeature] + ':' + str(bestBinarySplitVal)

        myTree = {bestFeatLabel: {}}

        # 拷贝防止labels被修改
        subLabels = labels[:]
        del subLabels[bestFeature]

        lSet, rSet = self.binsplitDataSet(dataSet, bestFeature, bestBinarySplitVal)

        #删除已选最佳特征的值
        lSet = delete(lSet, bestFeature, axis=1)
        rSet = delete(rSet, bestFeature, axis=1)

        #递归构建决策树
        myTree[bestFeatLabel]['<=' + str(round(float(bestBinarySplitVal), 3))] = self.createTree(rSet, subLabels, max_level - 1)
        myTree[bestFeatLabel]['>' + str(round(float(bestBinarySplitVal), 3))] = self.createTree(lSet, subLabels, max_level - 1)

        return myTree

    def labelsDict(self, labels):
        """
        把特征与索引建立字典
        :param labels:
        :return:
        """
        length = len(labels)
        predictLabelsDict = {}

        for i in range(length):
            currentLabel = labels[i]
            if currentLabel not in predictLabelsDict.keys():
                predictLabelsDict[currentLabel] = i

        return predictLabelsDict

    def subsample(self, trainDataSet):
        """
        样本有放回随机采样,输入数据集，返回采样子集'
        :param dataset: 数据集
        :return:随机抽样的后的数据集
        """
        row, col = shape(trainDataSet)
        sample = zeros(shape=(row, col))

        for addrow in range(row):
            index = randrange(row)
            sample[addrow] = trainDataSet[index, :]

        return sample

    def RondomForest(self, dataSet, labels, n, max_level):
        """
        随机森林
        :param dataSet: 数据集
        :param labels: 特征
        :param n: 决策树个数
        :param max_level: 决策树深度
        :return: 森林
        """
        Trees = []
        for i in range(n):
            dataSet = self.subsample(dataSet)
            # print dataSet
            tree = self.createTree(dataSet, labels, max_level)
            Trees.append(tree)

        return Trees

    def predict(self, tree, predictData, predictLabelsDict):
        """
        预测
        :param tree: 决策树
        :param predictData: 预测数据
        :param predictLabelsDict: 特征字典索引
        :return: 预测值
        """
        if type(tree) is not dict:
            return tree
        root=tree.keys()[0]
        #取出根节点，得到最优特征和阀值
        feature,threshold = root.split(':')
        threshold=float(threshold)
        #递归预测
        if predictData[predictLabelsDict[feature]]>threshold:
            return self.predict(tree[root]['>'+str(round(float(threshold),3))], predictData, predictLabelsDict)
        else:
            return self.predict(tree[root]['<='+str(round(float(threshold),3))], predictData, predictLabelsDict)

    def getPredictY(self, trees, predictDatas, predictLabelsDict):
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
                Y = self.predict(trees[i], samplerow, predictLabelsDict)
                predictAlltreeY.append(Y)
            predictAlltreeAvgY = sum(predictAlltreeY) / length
            predictY.append(predictAlltreeAvgY)

        return predictY

    def calRSS(self, y, predicty):
        """
        计算残差平方和
        :param trainDataSet: 数据集
        :return: 残差平方和
        """
        RSS = 0
        length = len(y)

        for i in range(length):
            RSS += (y[i] - predicty[i]) ** 2

        return RSS


    def result(self):
        """
        结果
        :return: Y
        """
        # # 得到数据集及特征
        # dataSet, labels = self.getDatasets(filename)
        #
        # #得到数据集的数组形式
        # dataSet = np.array(dataSet)
        #
        # # 得到训练集与测试集
        # trainData, predictData = self.trainDataAndPredictData(dataSet, self.selIndex)

        # 得到预测集y
        y = self.predictData[:, -1]

        # 产生森林 参数（训练集，特征，方差，树的个数，随机选择特征的个数，树的森度）
        Trees = self.RondomForest(self.trainData, self.labels, self.treeNum, self.treeDepth)

        # 预测 先把特征与索引建立字典为预测时找到对应的数据
        predictLabelsDict = self.labelsDict(self.labels)

        # 得到测试集每棵树的应变量的值 参数(森林，测试集，特征索引字典)
        predictY = self.getPredictY(Trees, self.predictData, predictLabelsDict)

        RSS = self.calRSS(y, predictY)

        return RSS


