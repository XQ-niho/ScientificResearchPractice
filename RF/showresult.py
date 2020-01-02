# coding=utf-8

import numpy as np
import time
from RF.OptRF import OPTRF
from RF.TraRF import TRARF
from RF.selectTrainAndPredictDataSets import GETDATASET

def main():
    path = "../datas/blogData.csv"

    treeNum = 50
    treeDepth = 15

    start = time.clock()

    gds = GETDATASET()

    TRAlabels, TRAtrainData, TRApredicData = gds.getTRADataSet(path)
    OPTlabels, OPTtrainData, OPTpredicData = gds.getOPTDataSet(path)
    print OPTlabels

    tra = TRARF(treeNum, treeDepth, TRAtrainData, TRApredicData, TRAlabels)
    RSS= tra.result()
    opt = OPTRF(treeNum, treeDepth, OPTtrainData, OPTpredicData, OPTlabels)
    optRSS = opt.result()

    average_error = np.sqrt(RSS / len(TRApredicData))
    opt_average_error = np.sqrt(optRSS / len(OPTpredicData))

    end = time.clock()

    print """
    尊敬的用户，建模结果如下：
    ------------------------
    建立模型耗时：{}s
    传统随机森林测试集残差平方和：{}
    改进随机森林测试集残差平方和：{}
    传统随机森林测试集平均相对误差：{}
    改进随机森林测试集平均相对误差：{}
    ------------------------
    训练集和测试集之比：7:3
    单颗树划分属性标准：方差
    叶子节点处理方式：取平均值
    """.format(end - start, RSS, optRSS,
               average_error, opt_average_error)

if __name__ == '__main__':
    main()
