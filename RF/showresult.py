# coding=utf-8

import numpy as np
from RF.regRF import RF
from RF.selectTrainAndPredictDataSets import GETDATASET

def regResult(path, treeNum, treeDepth):

    # treeNum = 40
    # treeDepth = 10


    gds = GETDATASET()

    TRAlabels, TRAtrainData, TRApredicData = gds.getTRADataSet(path)
    OPTlabels, OPTtrainData, OPTpredicData = gds.getOPTDataSet(path)

    tra = RF(treeNum, treeDepth, TRAtrainData, TRApredicData, TRAlabels)
    RSS= tra.result()
    opt = RF(treeNum, treeDepth, OPTtrainData, OPTpredicData, OPTlabels)
    optRSS = opt.result()

    average_error = np.sqrt(RSS / len(TRApredicData))
    opt_average_error = np.sqrt(optRSS / len(OPTpredicData))

    return RSS, optRSS, average_error, opt_average_error, OPTlabels
