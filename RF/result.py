# -*- coding:utf-8 -*-

import numpy as np
from numpy import *
from RF.regRF import RF
from RF.selectTrainAndPredictDataSets import GETDATASET

def regreResult(path, tree_min, tree_spa, tree_max, treeDepth):
    # path = "../datas/default_plus_chromatic_features_1059_tracks_1.csv"
    #
    # # treeNum = 2
    # treeDepth = 10

    gds = GETDATASET()

    TRAlabels, TRAtrainData, TRApredicData = gds.getTRADataSet(path)

    OPTlabels, OPTtrainData, OPTpredicData = gds.getOPTDataSet(path)

    TRARSSALL = []; OPTRSSALL = []
    for count in range(10):
        TRARSSL = []; OPTRSSL = []
        for treenum in range(tree_min, tree_max+1, tree_spa):
            #传统的随机森林
            tra = RF(treenum, treeDepth, TRAtrainData, TRApredicData, TRAlabels)
            TRARSS = tra.result()
            TRARSSL.append(TRARSS)
            #优化的随机森林
            opt = RF(treenum, treeDepth, OPTtrainData, OPTpredicData, OPTlabels)
            OPTRSS = opt.result()
            OPTRSSL.append(OPTRSS)
        TRARSSALL.append(TRARSSL)
        OPTRSSALL.append(OPTRSSL)

    TRARSSALL = np.array(TRARSSALL)
    OPTRSSALL = np.array(OPTRSSALL)

    cols = TRARSSALL.shape[1]
    averageTraRss = []
    averageOptRss = []

    for col in range(cols):
        averageTraRss.append(sum(TRARSSALL[:, col]) / cols)
        averageOptRss.append(sum(OPTRSSALL[:, col]) / cols)

    min_ave_optrss = min(averageOptRss)
    everyopt = np.sqrt(min_ave_optrss / len(OPTpredicData))

    ops = -1

    for index, value in enumerate(averageOptRss):
        if value == min_ave_optrss:
            ops = index
            break

    min_ave_trarss = averageTraRss[ops]
    everytra = np.sqrt(min_ave_trarss / len(TRApredicData))

    return OPTlabels, averageTraRss, averageOptRss, min_ave_optrss,\
           min_ave_trarss, everyopt, everytra