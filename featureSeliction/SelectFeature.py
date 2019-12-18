#-*- coding: utf-8 -*-
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from minepy import MINE
import copy


class ChooseFeature(object):
    def dCor(self,X, Y):
        '''
        计算X，Y的距离相关系数
        :param X: 特征
        :param Y: 类别或者特征
        :return:
        '''
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
            raise ValueError("number of samples must match")
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return dcor

    '''
    a = [1,2,3,4,5]
    b = np.array([1,2,9,4,4])
    print(dCor(a,b))
    '''

    def MIC(self,X, Y):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(X, Y)
        return mine.mic()

    def MCC(self,X, Y):
        X = np.atleast_1d(X)
        Y = np.atleast_1d(Y)
        return (self.MIC(X, Y) + self.dCor(X, Y)) / 2

    def calculate_x(self,data):
        value_x = list()
        for i in range(len(data)):
            value_x.append([])
            for j in range(len(data)):
                if i < j:
                    value_x[i].append(self.MCC(data[i], data[j]))
                if i == j:
                    value_x[i].append(1)
                if i > j:
                    value_x[i].append(value_x[j][i])
                #print("xxxx",value_x)
        return value_x

    def calculate_y(self,data,Y):
        value_y = list()
        for i in range(len(data)):
            value_y.append(self.MCC(data[i],Y))
        return value_y


    def read_values(self,data, list_header, local_number):
        # print("---------")
        '''
        将datafram 类型转换为 list类型
        :param data: 想要转换的数据文件
        :param list_header: 数据文件的第一行，及数据文件名
        :param local_number: 文件名的索引值
        :return:
        '''
        M = data[[list_header[local_number]]]
        data = M.values.T.tolist()
        change_values = map(float, data[0][:])
        # print(change_values)
        return change_values

    def first_feature(self,value_y,data_test, data_test_lable, data_feature, data_feature_lable, data_feature_order, tab, ):
        Mmax = float('-inf')
        '''
        选出第一个候选子集中的特征
        :param Y: 类别的组值
        :param list_header: 数据样本中特征的名称
        :param data_test: 原本的训练集数据
        :param data_left: 未被选择的特征集合列表
        :param data_feature: 被选择的特征集合列表
        :return: 创建了个有一个被选择的特征列表，跟新未被选择的特征剩余的特征集合列表
        '''
        first_feature_order = -1

        for i in range(len(value_y)):
            if value_y[i] > Mmax:
                Mmax = value_y[i]
                # print(Mmax)
                first_feature_order = i
        first_feature_values = data_test[first_feature_order]
        data_feature_lable.append(data_test_lable[first_feature_order])
        data_feature_order.append(first_feature_order)
        tab[first_feature_order] = 1
        data_feature.append(first_feature_values)
        # print(data_left[first_feature_order])
        # print(data_feature_lable)
        # print(data_left_lable[first_feature_order])
        # print (data_left_lable)
        return data_feature, data_feature_lable, tab, data_feature_order

    def other_feature(self,data_test, data_test_lable, value_y, value_x, data_feature, data_feature_lable,
                      data_feature_order,tab,):
        Mmax = float('-inf')
        '''
        选出剩下的最优特征
        :param Y: 类别的组值
        :param data_left: 未被选择的特征
        :param data_feature: 被选择的特征
        :return: 更新的被选择的特征，跟新未被选择的特征
        '''
        feature_order = -1

        for i in range(len(data_test)):
            if tab[i] == 1:
                continue
            # print("i",i)
            # print (data_left[i])
            MCC_left_y = value_y[i]
            MCC_left_feature = 0
            for j in data_feature_order:
                # print("len",len(data_feature))
                # print("j",j)
                MCC_left_feature = MCC_left_feature + value_x[i][j]
                #print("MCC_left_feature",MCC_left_feature)
            if MCC_left_y - ((MCC_left_feature) / len(data_feature)) > Mmax:
                # print("dfdghgfjgjhgj")
                Mmax = MCC_left_y - ((MCC_left_feature) / len(data_feature))
                feature_order = i
                # print(feature_order)
        if Mmax >= 0:
            data_feature_lable.append(data_test_lable[feature_order])
            data_feature.append(data_test[feature_order])
            tab[feature_order] = 1
            data_feature_order.append(feature_order)
            self.other_feature(data_test, data_test_lable, value_y, value_x, data_feature, data_feature_lable,
                          data_feature_order, tab, )
        else:
            return data_feature, data_feature_lable, data_feature_order

        #print("Mmax", Mmax)
        #print("value_y",value_y[feature_order])

        #del data_left[feature_order]
        #del data_left_lable[feature_order]


    def return_data(self,filename):
    #def return_data(self, feature_test,list_number,result_y,result_x,class_y,data_header):
        '''
        :param filename: 文件路径
        :param choosef_num: 特征选择数
        :return: 返回选择的数据形式为横着的列表
        '''
        train_test = pd.read_csv(filename)   #1
        print train_test
        train_test = train_test.iloc[:, 0:]  # 读出所有特征数据，数据类型为矩阵类型
        list_number = train_test.columns.values  # 原始特征值的标签
        class_y = self.read_values(train_test, list_number, local_number=-1)
        tab = [0]*len(list_number)
        feature_choose_values = list()  # 最终所存的特征值
        feature_choose_lable = list()  # 最终所存的特征标签
        feature_test = list()  #1
        feature_test_oder = list()
        #label = train_test[['animal']]#读出某一列的所有数据
        train_test = train_test.iloc[:, 0:-1]  # 读出所有特征数据，数据类型为矩阵类型  #1
        data_header = train_test.columns.values.tolist()  # 不含有类别属性的特征值标签表
        for i in range(len(data_header)):
            feature_test.append(self.read_values(train_test, data_header, i))
        result_y = self.calculate_y(feature_test,class_y)
        result_x = self.calculate_x(feature_test)
        #print("result_y",result_y)
        #print("result_x",result_x)
        self.first_feature(result_y,feature_test,data_header,feature_choose_values, feature_choose_lable,
                           feature_test_oder, tab)
        self.other_feature(feature_test, data_header, result_y, result_x, feature_choose_values, feature_choose_lable,
                       feature_test_oder, tab)
        feature_choose_values.append(class_y)
        # feature_choose_lable.append(list_number[-1])
        #print(feature_choose_lable),print(np.array(feature_choose_values).transpose())
        return np.array(feature_choose_values).transpose(),feature_choose_lable


if __name__ == "__main__":
    test1 = ChooseFeature()
    data,datalable = test1.return_data("../datas/CASP.csv")
    print data
    print datalable










