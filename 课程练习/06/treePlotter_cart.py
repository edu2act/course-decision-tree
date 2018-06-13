

'''{'index':index,'value',value,'left':left,'right':right}
字典的形式是 特征索引，特征取值，左子树，右子树
顺序要一样，键值的名称可以不一样'''
import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="round4", color='#3366FF')  #定义判断结点形态
leafNode = dict(boxstyle="circle", color='#FF6633')  #定义叶结点形态
arrow_args = dict(arrowstyle="<-", color='g')  #定义箭头

#绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


#计算叶结点数
def getNumLeafs(myTree):
    numLeafs = 0    
    secondDict=[myTree[list(myTree.keys())[2]],myTree[list(myTree.keys())[3]]]   ###++修改了

    for key in secondDict:
        if type(key).__name__ == 'dict':
            numLeafs += getNumLeafs(key)
        else:
            numLeafs += 1
    return numLeafs


#计算树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    # firstStr = list(myTree.keys())[0]
    # secondDict = myTree[firstStr]
    
    secondDict=[myTree[list(myTree.keys())[2]],myTree[list(myTree.keys())[3]]]   ###++修改了

    for key in secondDict:
        if type(key).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(key)
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


#在父子结点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr= str(myTree[list(myTree.keys())[0]])+' '+ str(myTree[list(myTree.keys())[1]])   ###++修改了(节点处的标签)
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, '')  #在父子结点间填充文本信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  #绘制带箭头的注释
    # secondDict = myTree[firstStr]
    secondDict=[myTree[list(myTree.keys())[2]],myTree[list(myTree.keys())[3]]]    ###++修改了
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict:                           ###++修改了
        if type(key).__name__ == 'dict':             ###++修改了
            plotTree(key, cntrPt, '')          ###++修改了
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(key, (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)###++修改了
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, '')
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()




