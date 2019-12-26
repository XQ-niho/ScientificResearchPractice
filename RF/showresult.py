# coding=utf-8

import numpy as np
import time
from RF.OptRF import OPTRF
from RF.TraRF import TRARF
from RF.selectTrainAndPredictDataSets import GETDATASET

def main():
    path = "../datas/data1.csv"

    treeNum = 10
    treeDepth = 3

    start = time.clock()

    gds = GETDATASET()

    TRAlabels, TRAtrainData, TRApredicData = gds.getTRADataSet(path)
    OPTlabels, OPTtrainData, OPTpredicData = gds.getOPTDataSet(path)

    tra = TRARF(treeNum, treeDepth, TRAtrainData, TRApredicData, TRAlabels)
    RSS, trainRSS = tra.result()
    opt = OPTRF(treeNum, treeDepth, OPTtrainData, OPTpredicData, OPTlabels)
    optRSS, optTrainRSS = opt.result()

    average_error = np.sqrt(RSS / len(TRApredicData))
    train_average_error = np.sqrt(trainRSS / len(TRAtrainData))
    opt_average_error = np.sqrt(optRSS / len(OPTpredicData))
    opt_train_average_errror = np.sqrt(optTrainRSS / len(OPTtrainData))

    end = time.clock()

    print """
    尊敬的用户，建模结果如下：
    ------------------------
    建立模型耗时：{}s
    传统随机森林训练集残差平方和：{}，改进随机森林训练集残差平方和：{}
    传统随机森林测试集残差平方和：{}，改进随机森林测试集残差平方和：{}
    传统随机森林训练集平均相对误差：{}，改进随机森林训练集平均相对误差：{}
    传统随机森林测试集平均相对误差：{}，改进随机森林测试集平均相对误差：{}
    ------------------------
    训练集和测试集之比：7:3
    单颗树划分属性标准：方差
    叶子节点处理方式：取平均值
    """.format(end - start, trainRSS, optTrainRSS, RSS, optRSS,
               train_average_error, opt_train_average_errror,
               average_error, opt_average_error)

if __name__ == '__main__':
    main()
