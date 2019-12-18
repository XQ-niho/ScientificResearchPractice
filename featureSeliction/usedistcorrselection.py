# -*- coding:utf-8 -*-
import csv

import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
from numpy import *
from minepy import MINE


class featureSelection():

    def __init__(self):
        pass

    def readDataSet(self,filename):
        """
        读取数据
        :param filename:文件路径
        :return:
        """
        labels = []
        dataSet = []
        dataSetS = []
        with open(filename) as file:
            curLien = file.readline()
            line1 = curLien.strip("\n").split("	")
            labels.extend(line1[:-1])
            for line in file.readlines():
                fields = line.strip("\n").split("	")
                dataSet.append(map(float, fields[:]))

        for i in range(len(labels)):
            data = [example[i] for example in dataSet]
            dataSetS.append(data)
        dependent_variable = [example[-1] for example in dataSet]
        dependent = line1[-1]

        return labels, dataSetS, dependent, dependent_variable

    def getDataSet(self,csvfilename):
        """
        读取csv文件
        :param csvfilename: 文件名或路径
        :return: 数据及特征
        """
        dataSet = []
        labels = []

        df = pd.read_csv(csvfilename)
        #读取第一行特征及因变量
        label = list(df.columns.values)
        #特征
        labels = label[: -1]
        #读取每个特征的特征值
        for i in range(len(labels)):
            #一列特征值
            value= df[[labels[i]]]
            #将一列转换为一行列表
            data = value.values.T.tolist()
            #把特征值转换为float
            dataValue = map(float, data[0][:])
            dataSet.append(dataValue)
        #因变量
        dependent = label[-1]
        #因变量值
        dependent_variable = df[[dependent]].values.T.tolist()
        dependent_value = map(float,dependent_variable[0][:])


        return labels, dataSet, dependent, dependent_value

    def calDistcorr(self,X, Y):
        """
        计算距离相关系数
        :param X: 自变量
        :param Y: 因变量
        :return: 距离相关系数
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        if np.prod(X.shape) == len(X):
            X = X[:, None]
        if np.prod(Y.shape) == len(Y):
            Y = Y[:, None]
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        n = X.shape[0]
        if Y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples must match')

        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

        return dcor

    def calMIC(self,X,Y):
        """
        计算最大信息系数
        :param X: 自变量
        :param Y: 因变量
        :return: 最大信息系数
        """
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)

        mine = MINE(alpha = 0.6, c = 15)
        mine.compute_score(X, Y)
        MIC = mine.mic()

        return MIC

    def getTheFirstFeature(self, dataSet, Y):
        """
        选择出第一个特征
        :param dataSet:
        :return: maxmdis:最大的mcc, featureId:第一个特征得索引
        """
        allmdis = []
        maxmdis = -1
        thefirstfeatureId = -1
        length = len(dataSet)
        for i in range(length):
            dis = self.calDistcorr(dataSet[i], Y)
            mic = self.calMIC(dataSet[i], Y)
            mdis = (mic + dis) / 2
            allmdis.append(mdis)
            if mdis > maxmdis:
                maxmdis = mdis
                thefirstfeatureId = i

        return maxmdis, thefirstfeatureId, allmdis


    def allFeaturesMCC(self, dataSet):
        """
        计算出两两特征之间的MCC
        :param dataSet:
        :return:
        """
        lenth = len(dataSet)

        all_featuresMCC = zeros(shape=(lenth, lenth))

        for i in range(lenth -1):
            for j in range(i+1,lenth):
                mccff = self.calDistcorr(dataSet[i], dataSet[j]) + self.calMIC(dataSet[i], dataSet[j])
                # print "[{0}][{1}:{2}]".format(i,j,mccff)
                all_featuresMCC[i][j] = mccff

        for n in range(lenth):
            for m in range(lenth):
                if all_featuresMCC[n][m] == 0:
                    all_featuresMCC[n][m] = all_featuresMCC[m][n]

        return all_featuresMCC

    def chooseBestFeature(self, dataSet, isChooseFeaturesIndex, isChooseFeaturesValue, mccFY, betweenFeatureMCC, y, m, cnt):
        """
        特征选择
        :param dataSet: 数据集
        :param isChooseFeaturesIndex: 已选特征索引
        :param isChooseFeaturesValue: 已选特征值
        :param mccFY: 所有特征与y的MCC
        :param betweenFeatureCorr: 特征之间的距离相关系数
        :param y: 因变量
        :param m: 选择特征的个数
        :param cnt: 控制器
        :return: 选择的特征索引及特征值
        """

        if len(isChooseFeaturesIndex) == m:
            return

        dataLen = len(dataSet)

        if m > dataLen:
            print u"特征个数为：{0}，不足：{1}个，请重新选择".format(len(dataSet),m)
            return

        if cnt == 1:
            maxmdis, thefirstfeatureId, allmdis = self.getTheFirstFeature(dataSet, y)
            isChooseFeaturesIndex.append(thefirstfeatureId)
            # print thefirstfeatureId
            isChooseFeaturesValue.append(dataSet[thefirstfeatureId])
            mccFY = allmdis

            cnt += 1

            self.chooseBestFeature(dataSet, isChooseFeaturesIndex, isChooseFeaturesValue, mccFY, betweenFeatureMCC, y, m, cnt)

        else:
            maxMCC = float("-inf")
            notChooseIndex = arange(dataLen)
            isChooselen = len(isChooseFeaturesIndex)
            nextAddIndex = -1

            for index in notChooseIndex:
                mccfy = 0; mccff = 0
                if index not in isChooseFeaturesIndex:
                    mccfy = mccFY[index]
                    for isChooseIndex in isChooseFeaturesIndex:
                        if isChooseIndex > index:
                            mccff += betweenFeatureMCC[index][isChooseIndex] / 2
                        else:
                            mccff += betweenFeatureMCC[isChooseIndex][index] / 2
                    if mccfy - mccff / isChooselen  >= maxMCC:
                        maxMCC = mccfy - mccff / isChooselen
                        nextAddIndex = index

            isChooseFeaturesIndex.append(nextAddIndex)
            isChooseFeaturesValue.append(dataSet[nextAddIndex])

            self.chooseBestFeature(dataSet, isChooseFeaturesIndex, isChooseFeaturesValue, mccFY, betweenFeatureMCC, y, m, cnt)

    def getBestFeatuleAanValue(self, path, m):
        """
        得到选出的特征及数据集
        :param path:数据集路径
        :param m: 要选择的特征个数
        :return: 特征，数据
        """
        labels, dataSet, Y, YValue  = self.getDataSet(path)

        isChooseFeaturesIndex = []
        isChooseFeaturesValue = []
        betweenFeatureMCC = self.allFeaturesMCC(dataSet)
        # print betweenFeatureMCC
        self.chooseBestFeature(dataSet, isChooseFeaturesIndex, isChooseFeaturesValue, [], betweenFeatureMCC, YValue, m, 1)
        # print isChooseFeaturesIndex
        # print isChooseFeaturesValue

        dataSets = []
        chooseFeature = []

        for i in range(len(isChooseFeaturesValue[0])):
            data = [example[i] for example in isChooseFeaturesValue]
            data.append(YValue[i])
            dataSets.append(data)
        for index in isChooseFeaturesIndex:
            chooseFeature.append(labels[index])


        return dataSets, chooseFeature

if __name__ =="__main__":
    path = "../datas/winequalityred.csv"
    fs = featureSelection()
    data, features = fs.getBestFeatuleAanValue(path, 4)
    print data
    print features