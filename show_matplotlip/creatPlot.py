# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # fc表示框填充色，0-1表示黑到白
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# plotTree函数
def plotMidText(cntrPt, parentPt, txtString):  # 在父子节点间填充文本信息
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)  # 在cntrPt和parentPt中点写一段文本，内容为txtString


# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType): # (节点的文字标注，节点中心位置，节点起点位置，节点属性)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]#第一个关键字是第一次划分数据集的类别标签
    secondDict = myTree[firstStr] #第一个关键字后的字典
    for key in secondDict.keys(): #从第一个关键字出发，遍历整棵树的所有子节点
        if type(secondDict[key]).__name__ == 'dict': #测试节点的数据类型是否为字典
            numLeafs += getNumLeafs(secondDict[key]) #是字典类型，则递归调用getNumLeafs
        else:
            numLeafs += 1 #非字典类型则为叶子节点
    return numLeafs

#获取树的层数
def getTreeDepth(myTree): #计算遍历过程中遇到判断节点的个数
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #若为非叶子节点，则递归计算树的深度
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotTree(myTree, parentPt, nodeTxt):  # 计算宽与高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]  # 第一个关键字是第一次划分数据集的类别标签
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)  # 其他非叶节点也用xOff算，但是不改变xOff值，只有真的画了一个叶节点，才改变xOff的值
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # （节点文本内容，子节点坐标，父节点坐标，节点属性）
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少y偏移，1.0/plotTree.totalD是层距
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 定义了一个框架，序号是1，背景色是白色
    fig.clf()
    axprops = dict(xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2],
                   yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0,1.2])  # 列表 xticks是x轴上将显示的坐标，yticks是y轴上将显示的坐标，空列表则不显示坐标
    createPlot.ax1 = plt.subplot(111, frameon=True, **axprops)  # 定义一个子图窗口（1行1列第1个窗口） 隐藏坐标轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 是决策树的叶子数，也代表宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 是决策树的深度
    plotTree.xOff = -0.5 / plotTree.totalW  # xOff代表的是刚刚画完的叶节点的x坐标，注意是叶节点
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()