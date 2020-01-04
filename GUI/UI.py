# -*- coding:utf-8 -*-

import pandas as pd
import wx
import wx.grid
import time
import os
import csv
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

# from ARRF_audiology import result, getAttr_grop
from RF import cart_of_featureval_continuity_sort_RF as sr
from RF import showresult as rr
from RF import result as re

label =[]

class TestFrame():

    def InitUI(self):
        self.frame = wx.Frame(parent=None, title = '基于随机森林的中医药数据分析系统',  size=(1200, 600), pos=(400, 120))
        p = wx.Panel(self.frame)
        icon = wx.EmptyIcon()
        icon.CopyFromBitmap(wx.BitmapFromImage(wx.Image(("../imgs/logo.png"), wx.BITMAP_TYPE_PNG)))
        self.frame.SetIcon(icon)

        #定义最大的盒子，竖直方向
        Max_V_Box = wx.BoxSizer(wx.VERTICAL)

        Top_H_Box = wx.BoxSizer()

        self.filename = wx.TextCtrl(p, size = (600,28))
        Top_H_Box.Add(self.filename, 0, wx.ALIGN_CENTER)
        openfile = wx.Button(p,label="打开文件")
        Top_H_Box.Add(openfile, 0,flag = wx.LEFT,border= 5)
        openfile.Bind(wx.EVT_BUTTON, self.OnOpenFile)

        group = wx.Button(p, label="数据分组")
        Top_H_Box.Add(group, 0, flag=wx.LEFT, border=5)
        group.Bind(wx.EVT_BUTTON, self.GroupData)#####################

        fordata = wx.Button(p,label="预测")
        Top_H_Box.Add(fordata, 0,flag = wx.LEFT,border= 5)
        fordata.Bind(wx.EVT_BUTTON, self.F_met)

        analydata = wx.Button(p,label="分析结果")
        Top_H_Box.Add(analydata, 0,flag = wx.LEFT,border= 5)
        sets = wx.Button(p,label="设置")
        Top_H_Box.Add(sets, 0,flag = wx.LEFT,border= 5)

        Max_V_Box.Add(Top_H_Box,0,  flag = wx.TOP|wx.LEFT,border= 5)

        Middle_H_Box = wx.BoxSizer()

        Left_V_Box =wx.BoxSizer(wx.VERTICAL)

        b1 = wx.Button(p,label = '常规')
        Left_V_Box.Add(b1,0)
        b2 = wx.Button(p, label='功能')
        Left_V_Box.Add(b2, 0, flag = wx.TOP|wx.BOTTOM,border= 5)
        t1 = wx.StaticText(p, label=u'随机森林')
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        t1.SetFont(font)
        Left_V_Box.Add(t1, 0,flag = wx.LEFT,border= 1)
        b3 = wx.Button(p, label='RF')
        Left_V_Box.Add(b3, 0,  flag = wx.TOP| wx.LEFT,border= 5)
        b4 = wx.Button(p, label='ARRF')
        b4.Bind(wx.EVT_BUTTON, self.Predict)
        Left_V_Box.Add(b4, 0, flag = wx.TOP| wx.LEFT,border= 5)
        b5 = wx.Button(p, label='FARF')
        Left_V_Box.Add(b5, 0, flag = wx.TOP|wx.BOTTOM| wx.LEFT,border= 5 )
        t2 = wx.StaticText(p, label=u'决策树')
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        t2.SetFont(font)
        Left_V_Box.Add(t2, 0, flag=wx.LEFT, border=1)
        b6 = wx.Button(p, label='CART')
        Left_V_Box.Add(b6, 0, flag = wx.TOP| wx.LEFT,border= 5)
        b7 = wx.Button(p, label='ID3')
        Left_V_Box.Add(b7, 0, flag = wx.TOP| wx.LEFT,border= 5)
        b8 = wx.Button(p, label='C4.5')
        Left_V_Box.Add(b8, 0, flag=wx.TOP | wx.LEFT|wx.BOTTOM, border=5)
        t2 = wx.StaticText(p, label=u'偏最小二乘')
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        t2.SetFont(font)
        Left_V_Box.Add(t2, 0, flag=wx.LEFT, border=1)
        b6 = wx.Button(p, label='B-PLS')
        Left_V_Box.Add(b6, 0, flag=wx.TOP | wx.LEFT, border=5)
        b7 = wx.Button(p, label='SOFT')
        Left_V_Box.Add(b7, 0, flag=wx.TOP | wx.LEFT, border=5)
        b8 = wx.Button(p, label='K-PLS')
        Left_V_Box.Add(b8, 0, flag=wx.TOP | wx.LEFT, border=5)
        #b7.SetBackgroundColour("#6495ED")



        Middle_H_Box.Add(Left_V_Box,flag = wx.ALL,border = 10)

        Right_Box = wx.BoxSizer()

        #self.grid.CreateGrid(15000, 1000)
        self.grid1 = wx.grid.Grid(p, style=wx.TE_MULTILINE | wx.TE_RICH2 | wx.HSCROLL)
        self.grid1.CreateGrid(50000, 1000)
        Right_Box.Add(self.grid1, 0, border=10)

        Middle_H_Box.Add(Right_Box,flag = wx.TOP , border = 10)

        Max_V_Box.Add(Middle_H_Box,0)

        p.SetSizer(Max_V_Box)

        self.frame.Show()
        self.frame.Centre()

    def OnOpenFile(self, event):
        result =[]
        self.dirname = ''
        filesFilter = 'All files(*.*)|*.*'
        dlg = wx.FileDialog(self.frame, "Choose a file", os.getcwd(), "", wildcard=filesFilter, style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:  # 读取文件，读一行写一行
            self.filepath = dlg.GetPath()
            self.filename.SetValue(self.filepath)
            with open(self.filepath) as file: # 打开csv文件
                csv_reader = csv.reader(file)# 将打开的文件装换成csv可读的对象
                global label
                label = list(next(csv_reader))
                #print(label)
                result.append(label)
                for each in csv_reader:
                    result.append(each)

            row = len(result)
            for i in range(row):
                res = result[i]
                #print(res)
                for j, value in enumerate(res):
                    self.grid1.SetCellValue(i, j, str(value))
            file.close()

    def GroupData(self, event):
        # dlg = BaseDialog(None, -1)
        dlg = AboutDialog()
        dlg.ShowModal()
        dlg.Destroy()

    def Predict(self,event):#####################################

        self.frame1 = wx.Frame(parent = None, title = '随机森林算法参数设置',  size=(800, 520), pos=(400, 120))
        self.p1 = wx.Panel(self.frame1)

        Box_sec = wx.BoxSizer(wx.VERTICAL)

        self.attr_grop, self.data = getAttr_grop(self.filepath)
        print('attr_grop:', self.attr_grop)
        # E, K = result(attr_grop, train)
        # print('平均误差为：', E)
        # print('kappa:', K)
        # train = data
        # min_size = 1
        # n_features = 70
        # max_depth = 50
        # n_trees = 40
        # class

        self.tc1 = wx.StaticText(self.p1,label = 'attr_group:')
        Box_sec.Add(self.tc1,0,wx.TOP|wx.LEFT,border = 5)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc1.SetFont(font)

        # self.area_text = wx.TextCtrl(self, -1, u'textArea多行文本，可Ctrl+A', size=(200, 50),
        #                              style=(wx.TE_MULTILINE | wx.TE_DONTWRAP))
        # self.area_text.SetInsertionPoint(0)
        # self.area_text.Bind(wx.EVT_KEY_UP, self.OnSelectAll)
        # box_sizer.Add(self.area_text)

        attr = ""
        for value in self.attr_grop:
            attr += (str(value)+"\n")
        self.tc2 = wx.TextCtrl(self.p1, value = attr,style=(wx.TE_MULTILINE | wx.TE_DONTWRAP|wx.TE_READONLY), size=(800, 75))
        # self.tc2 = wx.TextCtrl(self.p1,label = attr, size=(200, 50),
        #                               style=(wx.TE_MULTILINE | wx.TE_DONTWRAP))
        Box_sec.Add(self.tc2, 0)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc2.SetFont(font)

        global label
        Label = list(map(str, label))

        self.tt1 = wx.StaticText(self.p1, label='变量：', pos=(100, 100))

        self.val_all = wx.ListBox(self.p1, -1, choices=Label, style=wx.LB_EXTENDED, pos=(100, 120),
                                  size=(100, 300))  # wx.LB_EXTENDED | wx.LB_SORT多选并且排序
        self.val_all.Bind(wx.EVT_LISTBOX, self.on_list2, self.val_all)

        self.tt2 = wx.StaticText(self.p1,label = '自变量：', pos=(260, 100))

        self.val_top1 = wx.ListBox(self.p1, -1, pos=(260, 120), size=(100, 140))

        self.tt3 = wx.StaticText(self.p1, label='因变量：', pos=(260, 260))

        self.val_bottom1 = wx.ListBox(self.p1, -1, pos=(260, 280), size=(100, 140))

        btn1 = wx.Button(self.p1, label='->', pos=(210, 170), size=(30, 30))
        btn1.Bind(wx.EVT_BUTTON, self.btn1top)

        btn2 = wx.Button(self.p1, label='->', pos=(210, 330), size=(30, 30))
        btn2.Bind(wx.EVT_BUTTON, self.btn2top)

        # btn3 = wx.Button(self.p1, label='开始计算', pos=(130, 440), size=(100, 35))
        # btn3.Bind(wx.EVT_BUTTON, self.btn3rec)

        self.tt4 = wx.StaticText(self.p1, label='min_size：', pos=(383, 173), style = wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tt4.SetFont(font)
        self.min_size = wx.TextCtrl(self.p1, size=(120, 22),pos=(485, 170))

        self.tt5 = wx.StaticText(self.p1, label='n_features：', pos=(383, 213), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tt5.SetFont(font)
        self.n_features = wx.TextCtrl(self.p1, size=(120, 22), pos=(485, 210))

        self.tt6 = wx.StaticText(self.p1, label='max_depth：', pos=(383, 253), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tt6.SetFont(font)
        self.max_depth = wx.TextCtrl(self.p1, size=(120, 22), pos=(485, 250))

        self.tt7 = wx.StaticText(self.p1, label='n_trees：', pos=(383, 293), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tt7.SetFont(font)
        self.n_trees = wx.TextCtrl(self.p1, size=(120, 22), pos=(485, 290))

        btn3 = wx.Button(self.p1, label='开始建模', pos=(383, 330), size=(225, 30))
        btn3.Bind(wx.EVT_BUTTON, self.btn3rec)

        self.p1.SetSizer(Box_sec)

        self.frame1.Show()
        self.frame1.Centre()

    def Getvalue(self):
        return int(self.min_size.GetValue()),int(self.n_features.GetValue()),int(self.max_depth.GetValue()),int(self.n_trees.GetValue())

    def OnClick(self,event):

        print('参数1：',self.val1.GetValue())
        print('参数2：',self.val2.GetValue())

    def on_list2(self,event):
        self.listbox2 = event.GetEventObject()
        ##print(self.listbox2)
        #print(type(listbox2.GetSelections()))
        ##print('listbox2:' + str(self.listbox2.GetSelections()))

    def btn1top(self,event):
        self.val_top1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel1 = []
        global label
        #print(label)
        for i in f1:
            self.sel1.append(label[i])
        #print(self.sel1)
        self.val_top = wx.ListBox(self.p1 ,-1,choices = self.sel1 ,style=wx.LB_EXTENDED, pos=(260, 120), size=(100, 140))

    def btn2top(self,event):
        self.val_bottom1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel2 = []
        global label
        #print(label)
        for i in f1:
            self.sel2.append(label[i])
        #print(self.sel2)
        self.val_bottom = wx.ListBox(self.p1, -1,choices = self.sel2 ,style=wx.LB_EXTENDED, pos=(260, 280),size = (100,140))

    def btn3rec(self,event):

        min_size, n_features, max_depth, n_trees = self.Getvalue()#1,70,50,40
        E, K = result(self.attr_grop, self.data, min_size, n_features, max_depth, n_trees)
        print(E)
        print(K)

        self.frame2 = wx.Frame(parent=None, title='结果展示', size=(800, 330))
        self.p2 = wx.Panel(self.frame2)

        Box_ms = wx.BoxSizer(wx.VERTICAL)

        self.tc1 = wx.StaticText(self.p2, label='E:')
        Box_ms.Add(self.tc1, 0, wx.TOP | wx.LEFT, border=5)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc1.SetFont(font)

        #EaK = (str(E)+'\n'+str(K))
        self.tc2 = wx.TextCtrl(self.p2, value=(str(E)), style=(wx.TE_MULTILINE | wx.TE_DONTWRAP | wx.TE_READONLY),
                               size=(800, 60))
        Box_ms.Add(self.tc2, 0)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc2.SetFont(font)

        self.tc3 = wx.StaticText(self.p2, label='K:')
        Box_ms.Add(self.tc3, 0, wx.TOP | wx.LEFT, border=5)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc3.SetFont(font)

        self.tc4 = wx.TextCtrl(self.p2, value=(str(K)), style=(wx.TE_MULTILINE | wx.TE_DONTWRAP | wx.TE_READONLY),
                               size=(800, 60))
        Box_ms.Add(self.tc4, 0)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.tc4.SetFont(font)


        self.p2.SetSizer(Box_ms)

        self.frame2.Show()
        self.frame2.Centre()

    def F_met(self,event):#####################################

        self.frame_fmet = wx.Frame(parent = None, title = '参数设置',  size=(850, 500), pos=(400, 120))
        self.pfmet = wx.Panel(self.frame_fmet)

        Box_Fmet = wx.BoxSizer(wx.VERTICAL)

        global label
        Label = list(map(str, label))

        self.tt1 = wx.StaticText(self.pfmet, label='变量：', pos=(30, 50))

        self.val_all = wx.ListBox(self.pfmet, -1, choices=Label, style=wx.LB_EXTENDED, pos=(30, 70),
                                  size=(100, 300))  # wx.LB_EXTENDED | wx.LB_SORT多选并且排序
        self.val_all.Bind(wx.EVT_LISTBOX, self.on_list2_fmet, self.val_all)

        self.tt2 = wx.StaticText(self.pfmet, label='自变量：', pos=(190, 50))

        self.val_top1 = wx.ListBox(self.pfmet, -1, pos=(220, 70), size=(100, 140))

        self.tt3 = wx.StaticText(self.pfmet, label='因变量：', pos=(220, 210))

        self.val_bottom1 = wx.ListBox(self.pfmet, -1, pos=(220, 230), size=(100, 140))

        btn1 = wx.Button(self.pfmet, label='->', pos=(170, 120), size=(30, 30))
        btn1.Bind(wx.EVT_BUTTON, self.btn1top_fmet)

        btn2 = wx.Button(self.pfmet, label='->', pos=(170, 280), size=(30, 30))
        btn2.Bind(wx.EVT_BUTTON, self.btn2top_fmet)

        self.statictext = wx.StaticText(self.pfmet, label='select_forecast：', pos=(343, 123))
        list2 = ['分类预测', '回归预测']
        self.ch2 = wx.Choice(self.pfmet, -1, choices=list2, size=(240, 22), pos=(440, 118))
        self.ch2.Bind(wx.EVT_CHOICE, self.on_choice)

        self.ft1 = wx.StaticText(self.pfmet, label='tree_min：', pos=(343, 153), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.ft1.SetFont(font)
        self.tree_min = wx.TextCtrl(self.pfmet, size=(30, 22), pos=(440, 148), style=0)

        self.ft11 = wx.StaticText(self.pfmet, label='tree_spa:', pos=(472, 153), style=wx.ALIGN_CENTER)
        self.ft11.SetFont(font)
        self.tree_spa= wx.TextCtrl(self.pfmet, size=(30,22), pos=(550, 148), style=0)

        self.ft12 = wx.StaticText(self.pfmet, label='tree_max:', pos=(582, 153), style=wx.ALIGN_CENTER)
        self.ft12.SetFont(font)
        self.tree_max = wx.TextCtrl(self.pfmet, size=(30, 22), pos=(655, 148), style=0)

        self.ft2 = wx.StaticText(self.pfmet, label='tree_depth:', pos=(343, 183), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.ft2.SetFont(font)
        self.tree_depth = wx.TextCtrl(self.pfmet, size=(240, 22), pos=(440, 178))

        self.ft3 = wx.StaticText(self.pfmet, label='ratio:', pos=(343, 213), style=wx.ALIGN_CENTER)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        self.ft3.SetFont(font)
        self.ratio = wx.TextCtrl(self.pfmet,  value="7:3", size=(240, 22), pos=(440, 208), style=wx.TE_READONLY)

        btn3 = wx.Button(self.pfmet, label='开始建模', pos=(400, 243), size=(240, 30))
        btn3.Bind(wx.EVT_BUTTON, self.btn3rec_fmet)

        self.pfmet.SetSizer(Box_Fmet)

        self.frame_fmet.Show()
        self.frame_fmet.Centre()

    def on_choice(self,event):
        self.choose = event.GetString()

    def getSortParam(self):
        return int(self.tree_min.GetValue()), int(self.tree_spa.GetValue()), \
               int(self.tree_max.GetValue()), int(self.tree_depth.GetValue())

    def btn3rec_fmet(self,event):
        y1 = []
        y2 = []
        self.corr = ''
        self.train_corr = ''

        tree_min, tree_spa, tree_max, tree_depth = self.getSortParam()

        if self.choose == u'分类预测':
            start = time.clock()
            corrs, train_corrs = sr.sortResult(self.filepath, tree_min, tree_max, tree_spa, tree_depth)
            end = time.clock()

            best_trees = 0
            best_train_trees = 0
            max_train_corr = max(train_corrs)
            max_corr = max(corrs)
            for index, value in enumerate(corrs):
                if max_corr == value:
                    best_trees = index
                    break
            for index, value in enumerate(train_corrs):
                if max_train_corr == value:
                    best_train_trees = index
                    break

            # print self.corr
            self.frame2 = wx.Frame(parent=None, title='随机森林分类建模结果', size=(800, 600))
            y1 = corrs
            y2 = train_corrs
            self.str_corr = """
            尊敬的用户，分类建模最好时结果如下：
            ---------------------
            建模所耗时间：{}s
            最好时训练集决策树：{}
            此时训练集分类准确率：{}
            最好时测试集决策树：{}
            此时测试集分类准确率：{}
            ---------------------
            决策树的深度：{}
            训练集和测试集之比：7:3
            单颗树划分属性标准：信息增益
            叶子节点处理方式：投票选择
            """.format(end - start, tree_min+best_train_trees*tree_spa,
                       max_train_corr, tree_min+best_trees*tree_spa, max_corr, tree_depth)
        else:
            start = time.clock()

            labels, averageTraRss, averageOptRss, min_ave_optrss, \
            min_ave_trarss, everyopt, everytra = \
                re.regreResult(self.filepath, tree_min,
                        tree_spa, tree_max, tree_depth)

            end = time.clock()
            self.frame2 = wx.Frame(parent=None, title='随机森林改进建模对比结果', size=(800, 600))

            y1 = averageTraRss
            y2 = averageOptRss
            self.str_corr = """
            尊敬的用户，回归建模结果如下：
            ---------------------------
            建立模型耗时：{}s
            传统随机森林测试集残差平方和：{}
            改进随机森林测试集残差平方和：{}
            传统随机森林测试集平均相对误差：{}
            改进随机森林测试集平均相对误差：{}
            ---------------------------
            决策树深度：{}
            训练集和测试集之比：7:3
            单颗树划分属性标准：方差
            叶子节点处理方式：取平均值
            """.format(end - start, min_ave_trarss, min_ave_optrss,
                            everytra, everyopt, tree_depth)

        p2 = wx.Panel(self.frame2)

        Box_ms = wx.BoxSizer(wx.VERTICAL)

        tc1 = wx.StaticText(p2, label='建模结果:')
        Box_ms.Add(tc1, 0, wx.TOP | wx.LEFT, border=5)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        tc1.SetFont(font)

        tc2 = wx.TextCtrl(p2, value=self.str_corr, style=(wx.TE_MULTILINE | wx.TE_DONTWRAP | wx.TE_READONLY),
                                   size=(800, 200))
        Box_ms.Add(tc2, 0)
        font = wx.Font(11, wx.DECORATIVE, wx.NORMAL, wx.NORMAL)
        tc2.SetFont(font)

        figure = Figure()
        axes = figure.add_subplot(111)
        canvas = FigureCanvas(p2, -1, figure)
        Box_ms.Add(canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        p2.Fit()

        x = []
        for i in range(tree_min, tree_max + 1, tree_spa):
            x.append(i)

        axes.plot(x, y1, label='tra')
        axes.plot(x, y2, color='red', label='opt')

        p2.SetSizer(Box_ms)

        self.frame2.Show()
        self.frame2.Centre()


    def on_list2_fmet(self,event):
        self.listbox2 = event.GetEventObject()

    def btn1top_fmet(self, event):
        self.val_top1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel1 = []
        global label
        # print(label)
        for i in f1:
            self.sel1.append(label[i])
        # print(self.sel1)
        self.val_top = wx.ListBox(self.pfmet, -1, choices=self.sel1, style=wx.LB_EXTENDED, pos=(220, 70), size=(100, 140))

    def btn2top_fmet(self, event):
        self.val_bottom1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel2 = []
        global label
        # print(label)
        for i in f1:
            self.sel2.append(label[i])
        # print(self.sel2)
        self.val_bottom = wx.ListBox(self.pfmet, -1, choices=self.sel2, style=wx.LB_EXTENDED, pos=(220, 230),
                                     size=(100, 140))


class AboutDialog(wx.Dialog):

    def __init__(self):
        wx.Dialog.__init__(self, parent=None, id=200, title='计算提升度',size=(500,500))
        global label
        Label = list(map(str,label))

        self.val_all = wx.ListBox(self, -1, choices = Label,style=wx.LB_EXTENDED ,pos =(100,50), size =(100,300))  # wx.LB_EXTENDED | wx.LB_SORT多选并且排序
        self.Bind(wx.EVT_LISTBOX, self.on_list2, self.val_all)

        self.val_top1 = wx.ListBox(self,-1, pos=(260, 50),size = (100,140))

        self.val_bottom1 = wx.ListBox(self, -1,pos=(260, 210),size = (100,140))

        btn1 = wx.Button(self, label='->', pos=(210, 100), size=(30, 30))
        btn1.Bind(wx.EVT_BUTTON,self.btn1top)

        btn2 = wx.Button(self, label='->', pos=(210, 260), size=(30, 30))
        btn2.Bind(wx.EVT_BUTTON, self.btn2top)

        btn3 = wx.Button(self, label='开始计算', pos=(180, 370), size=(100,35))
        btn3.Bind(wx.EVT_BUTTON, self.btn3rec)

#获取参数值
    def Getvalue1(self):
        return self.val1.GetValue()

    def Getvalue2(self):
        return self.val2.GetValue()
#打印参数
    def OnClick(self,event):

        print('参数1：',self.val1.GetValue())
        print('参数2：',self.val2.GetValue())

    def on_list2(self,event):
        self.listbox2 = event.GetEventObject()
        print(self.listbox2)
        #print(type(listbox2.GetSelections()))
        print('listbox2:' + str(self.listbox2.GetSelections()))

    def btn1top(self,event):
        self.val_top1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel1 = []
        global label
        #print(label)
        for i in f1:
            self.sel1.append(label[i])
        print(self.sel1)
        self.val_top = wx.ListBox(self,-1,choices = self.sel1 ,style=wx.LB_EXTENDED, pos=(260, 50), size=(100, 140))

    def btn2top(self,event):
        self.val_bottom1.Destroy()
        f1 = self.listbox2.GetSelections()
        self.sel2 = []
        global label
        #print(label)
        for i in f1:
            self.sel2.append(label[i])
        print(self.sel2)
        self.val_bottom = wx.ListBox(self,-1,choices = self.sel2 ,style=wx.LB_EXTENDED, pos=(260, 210),size = (100,140))

    def btn3rec(self,event):
        print('自变量：'+str(self.sel1))
        print('因变量：'+str(self.sel2))



if __name__ == '__main__':
    app = wx.App(0)
    frame = TestFrame()
    frame.InitUI()
    app.MainLoop()