# -*- coding:utf-8 -*-

import pandas as pd

from RF.OptRF import OPTRF
from RF.TraRF import TRARF
from RF.selectTrainAndPredictDataSets import GETDATASET

def main():


    path = "../datas/winequalityred.csv"

    # treeNum = 2
    treeDepth = 10

    gds = GETDATASET()

    TRAlabels, TRAtrainData, TRApredicData = gds.getTRADataSet(path)

    OPTlabels, OPTtrainData, OPTpredicData = gds.getOPTDataSet(path)

    TRARSSALL = []; OPTRSSALL = []
    for count in range(10):
        TRARSSL = []; OPTRSSL = []
        for treenum in range(5, 100, 5):
            #传统的随机森林
            tra = TRARF(treenum, treeDepth, TRAtrainData, TRApredicData, TRAlabels)
            TRARSS = tra.result()
            TRARSSL.append(TRARSS)
            #优化的随机森林
            opt = OPTRF(treenum, treeDepth, OPTtrainData, OPTpredicData, OPTlabels)
            OPTRSS = opt.result()
            OPTRSSL.append(OPTRSS)
        TRARSSALL.append(TRARSSL)
        OPTRSSALL.append(OPTRSSL)


    df = pd.DataFrame(TRARSSALL, columns=['5','10','15','20','25','30','35','40','45','50','55','60','65','70',
                                          '75','80','85','90','95','100'])
    df.to_excel("../datas/WinequalityredTRA04.xlsx", index=False)

    dfopt = pd.DataFrame(OPTRSSALL, columns=['5','10','15','20','25','30','35','40','45','50','55','60','65','70',
                                          '75','80','85','90','95','100'])
    dfopt.to_excel("../datas/WinequalityredOPT04.xlsx", index=False)


if __name__ == "__main__":
    main()