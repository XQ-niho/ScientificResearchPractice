# -*-coding:utf-8 -*-
import csv

from numpy import *
import numpy as np

from featureSeliction.SelectFeature import ChooseFeature
from featureSeliction.usedistcorrselection import featureSelection

class GETDATASET():

    # def __init__(self, featureChoNum):
    #     self.featureChoNum = featureChoNum

    def OPTgetDataSet(self, filename):

        fs = ChooseFeature()

        # 得到特征选取后的数据 参数（文件路径，要选择的特征数）
        dataSet, labels = fs.return_data(filename)
        # print "{0}\n{1}".format(dataSet, labels)
        # labels, dataSet = selDataSetRes(filename, self.featureChoNum)
        # print "{0}\n{1}".format(dataSet, labels)

        #得到数据集的数组形式
        # dataSet = np.array(dataSet)

        return dataSet, labels

    def OPTtrainDataAndPredictData(self, dataSets, selIndex):
        """
        把数据按7：3分成训练集与测试集
        :param dataSet:
        :return:
        """
        dataSet = dataSets.tolist()

        length = len(dataSet)

        # # 产生长度为length的序列
        # array = arange(length)
        # # 将序列随机排列
        # random.shuffle(array)

        array = selIndex

        OPTpredictData = []
        OPTtrainData = []

        for index in array:
            if len(OPTtrainData) <= length * 0.7:
                OPTtrainData.append(dataSet[index])
            else:
                OPTpredictData.append(dataSet[index])

        OPTtrainData = np.array(OPTtrainData)
        OPTpredictData = np.array(OPTpredictData)

        return OPTtrainData, OPTpredictData

    def TRAgetDatasets(self, csvfilename):
        """
        获得数据集
        :param csvfilename: 文件名或文件路径
        :return: 数据集，特征
        """
        dataSets = []
        labels = []

        with open(csvfilename) as csvfile:
            # 读取csv文件
            csv_reader = csv.reader(csvfile)
            # 读取第一行
            label = next(csv_reader)
            # 获得特征
            labels = label[: -1]

            for row in csv_reader:
                # 把所有元素转换为float
                ro = map(float, row)
                dataSets.append(ro)

        # 得到数据集的数组形式
        dataSet = np.array(dataSets)

        return dataSet, labels

    def TRAtrainDataAndPredictData(self, dataSets):
        """
        把数据按7：3分成训练集与测试集
        :param dataSet:
        :return:
        """
        dataSet = dataSets.tolist()

        length = len(dataSet)

        # 产生长度为length的序列
        array = arange(length)
        # 将序列随机排列
        random.shuffle(array)
        # array = selIndex

        TRApredictData = []
        TRAtrainData = []

        for index in array:
            if len(TRAtrainData) <= length * 0.7:
                TRAtrainData.append(dataSet[index])
            else:
                TRApredictData.append(dataSet[index])

        TRAtrainData = np.array(TRAtrainData)
        TRApredictData = np.array(TRApredictData)

        return TRAtrainData, TRApredictData, array

    def getTRADataSet(self,filename):
        """
        得到传统随机森林的训练集与预测集
        :param filename:
        :return:
        """
        TRAdatas, TRAlabels = self.TRAgetDatasets(filename)
        TRAtrainData, TRApredicData, self.selIndex = self.TRAtrainDataAndPredictData(TRAdatas)

        return TRAlabels, TRAtrainData, TRApredicData

    def getOPTDataSet(self, filename):
        """
        得到优化随机森林的训练集与预测集
        :param filename:
        :return:
        """

        OPTdatas, OPTlabels = self.OPTgetDataSet(filename)
        OPTtrainData, OPTpredicData = self.OPTtrainDataAndPredictData(OPTdatas, self.selIndex)

        return OPTlabels, OPTtrainData, OPTpredicData


