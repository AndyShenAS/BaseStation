from numpy import *
from xlwt import Workbook, Formula
import xlrd
import csv
import pandas as pd
import chardet
import codecs  
import collections
import numpy
import  scipy.stats as stat
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle


###########################

# reader = pd.read_csv("SHX1_LTE-R02_CellPerformanceHours_20171016_1 (copy).CSV",chunksize=10,iterator=True)
# temp=reader.get_chunk()
# while temp is not None:
#     print(temp)
#     temp=reader.get_chunk()



#>>> df = pd.DataFrame(
#...         {'col1': [1, 2], 'col2': [0.5, 0.75]})
#>>> df
#   col1  col2
#0     1  0.50
#1     2  0.75


#>>> df.to_dict('records')
#[{'col1': 1.0, 'col2': 0.5}, {'col1': 2.0, 'col2': 0.75}]


# In [177]: reader = pd.read_table('tmp.sv', sep='|', chunksize=4)
# 
# In [178]: reader
# Out[178]: <pandas.io.parsers.TextFileReader at 0x1201e8518>
# 
# In [179]: for chunk in reader:
#    .....:     print(chunk)
#    .....: 
#    Unnamed: 0         0         1         2         3
# 0           0  0.469112 -0.282863 -1.509059 -1.135632
# 1           1  1.212112 -0.173215  0.119209 -1.044236
# 2           2 -0.861849 -2.104569 -0.494929  1.071804
# 3           3  0.721555 -0.706771 -1.039575  0.271860
#    Unnamed: 0         0         1         2         3
# 4           4 -0.424972  0.567020  0.276232 -1.087401
# 5           5 -0.673690  0.113648 -1.478427  0.524988
# 6           6  0.404705  0.577046 -1.715002 -1.039268
# 7           7 -0.370647 -1.157892 -1.344312  0.844885
#    Unnamed: 0         0        1         2         3
# 8           8  1.075770 -0.10905  1.643563 -1.469388
# 9           9  0.357021 -0.67460 -1.776904 -0.968914

###########################



srcPath = ['SHX1_LTE-R02_CellPerformanceHours_20171016_1 (copy).CSV','SHX2_LTE-R02_CellPerformanceHours_20171016_2 (copy).CSV','SHX4_LTE-R02_CellPerformanceHours_20171016_4 (copy).CSV']  
#srcPath = ['AI-碑林.CSV'] 

queryLocation = 'SHX_LTE-R01_CellResources_20171016 (copy).CSV'

# rows = []
# for src in srcPath:
#     with open(src,"r", encoding='utf-8') as csvfile:
#         data = csv.DictReader(csvfile)
#         tempRows = [row for row in data]
#     rows.extend(tempRows)


rows = []
for src in srcPath:
    reader = pd.read_csv(src,chunksize=1000,iterator=True)
    for chunk in reader:
        temp = chunk.to_dict('records')
        rows.extend(temp)
    print(src)



#print(rows[0:5])
keys=[]
for k,v in rows[0].items():
    keys.append(k)
# print(keys)
CGIs = []  #获取全部不重复的CGI值
for row in rows:
    temp = row['CGI']
    if (temp is not '') and (temp not in CGIs):
        CGIs.append(temp)

print('the length of CGIs is:  %d' % len(CGIs))
formedRows = collections.OrderedDict() #根据CGI键排序存储数据，key-CGI值；value-除去前六个字段的的其他字段值，用字典存放，每个字段有24个值，用list存放

#formedRows = {'cgi1':{'RRC连接建立成功率':['0.xx',....],'':[],....},'cgi2':{},....}
#列表，字典，元组

for CGI in CGIs:
    attrValues = collections.OrderedDict()
    for key in keys[6:]:
        attrValues[key] = []
    formedRows[CGI] = attrValues
for row in rows:
    for key in keys[6:]:
        if (row[key] is not '') and (row[key] is not None):
            formedRows[row['CGI']][key].append(row[key])

rows = []

print('formedRows ready......')


meanValues = []
#count = 0 

#eachCGI = [0.7,...]
#[[,,nan，],
#[],
#[].....
#]

for CGI in CGIs:  #算每个CGI的每个字段24小时均值，没有的字段用其他CGI对应字段的均值替代
    eachCGI = []
    #count += 1
    #if count < 6 : print(CGI)
    for k,v in formedRows[CGI].items():
        if v is not []:
            result = [float(item) for item in v]
            eachCGI.append(sum(result)/len(v))
            #eachCGI.append(None)
            #if count < 6 : print(v)
        else: 
            eachCGI.append(NaN)
    meanValues.append(eachCGI)
print('meanValues ready......')
#没有的字段用其他CGI对应字段的均值替代
globalMean = array(meanValues)
#Obtain mean of columns as you need, nanmean is just convenient.
col_mean = numpy.nanmean(globalMean,axis=0)

#Find indicies that you need to replace
inds = numpy.where(numpy.isnan(globalMean))

#Place column means in the indices. Align the arrays using take
globalMean[inds]=numpy.take(col_mean,inds[1])
#globalMean[0:5]

print('PCA begin......')
#数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
globalMean_minmax = min_max_scaler.fit_transform(globalMean)
#PCA 降维
pca = PCA(n_components=0.95)
pca.fit(globalMean_minmax)
newData_PCA = pca.transform(globalMean_minmax)
print('PCA result......')
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
# print(len(keys[6:]))

# print(newData_PCA[0:5])




#fig = plt.figure()
#ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#plt.scatter(newData_PCA[:, 0], newData_PCA[:, 1], newData_PCA[:, 2],marker='o')
#plt.show()

            





#MeanShift聚类
# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using

print('MeanShift begin......')

bandwidth = estimate_bandwidth(newData_PCA, quantile=0.06)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(newData_PCA)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = numpy.unique(labels)
n_clusters_ = len(labels_unique)

print('MeanShift result......')
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
print("number of estimated clusters : %d" % n_clusters_)
print(labels_unique)
#print(newData_PCA[0:5])
#print(newData_PCA[0, 0], newData_PCA[0, 1], newData_PCA[0, 2])
print(labels)
for k, col in zip(range(n_clusters_), colors):
    print(k, col)


# #############################################################################
# Plot result


#newData_PCA = 
#[[],
#[],
#[]....
#]

#numpy->array
#my_members=
#[true,
#false,
#...
#]





fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k   # my_members = [False  True False ..., False False False]
    cluster_center = cluster_centers[k]
    #plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    print('flag1')
    plt.scatter(newData_PCA[my_members, 0], newData_PCA[my_members, 1], newData_PCA[my_members, 2],c=col,marker='.')
    print('flag2')
    print("cluster : %d" % k)
    clusterNum = numpy.sum(my_members)
    #np.count_nonzero(my_members)
    print("total number is : %d" % clusterNum)
    #print(my_members)
    #print(newData_PCA[my_members, 0], newData_PCA[my_members, 1], newData_PCA[my_members, 2])
    plt.scatter(cluster_center[0], cluster_center[1], cluster_center[2],marker='o', c=col, edgecolors='k',)
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()









# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     #plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.scatter(newData_PCA[my_members, 0], newData_PCA[my_members, 1], newData_PCA[my_members, 2],c=col,marker='.')
#     plt.scatter(cluster_center[0], cluster_center[1], cluster_center[2],marker='o', c=col, edgecolors='k',)
# #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
# #              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()


# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(newData_PCA[:, 0], newData_PCA[:, 1], newData_PCA[:, 2],marker='o')
# plt.show()








#下面是生成地图显示的数据







queryLocation = ['SHX_LTE-R01_CellResources_20171016 (copy).CSV']


rows = []
for src in queryLocation:
    reader = pd.read_csv(src,chunksize=1000,iterator=True)
    for chunk in reader:
        temp = chunk.to_dict('records')
        rows.extend(temp)





keys=[]
for k,v in rows[0].items():
    keys.append(k)
# print(keys)
newCGIs = []  #获取全部不重复的CGI值
for row in rows:
    temp = row['CGI']
    if not temp in newCGIs:
         if temp is not '':
             newCGIs.append(temp)
#print(CGIs)
formedRows = collections.OrderedDict() #这里面存放了所有可供查询的CGI的经纬度值

for CGI in newCGIs:
    attrValues = collections.OrderedDict()
    for key in keys:
        attrValues[key] = ''
    formedRows[CGI] = attrValues

for row in rows:
    for key in keys:
        if (row[key] is not '') and (row[key] is not None):
            formedRows[row['CGI']][key] = row[key]


count = 0
for k,v in formedRows['460-00-590495-3'].items():
    print(k,v)



dicData = {}


for i in labels:
    dicData[i] = []

mapcolors = ['#D340C3','#0000FF','#DC143C','#1E90FF','#00BFFF','#00FFFF','#3CB371','#FFD700','#800000','#000000']

count = 0
for CGI in CGIs:
    if CGI in formedRows.keys():
        tempData = []
        tempData.append(formedRows[CGI]['经度'])
        tempData.append(formedRows[CGI]['纬度'])
        tempData.append(mapcolors[labels[count]])
        tempData.append(labels[count])
        tempData.append(CGI)
        dicData[labels[count]].append(tempData)
    count += 1


#var data = [{classify:[[,,,],[]...]}]
# print(data[0:5])
#dicData = {}
#dicData["data"] = data
#print(dicData)


#os.mknod('data.json') #创建空文件
#with open('data.json',"rw", encoding='utf-8') as jsonfile:
#    jsonfile.write('var data = '+dicData)


file = open('posData.js',"w", encoding='utf-8')

file.write('var data = '+str(dicData))
# 关闭
file.close()

print('end......')









# >>> a = {'abc':[1,2,3]}
# >>> a['abc']
# [1, 2, 3]
# >>> a['abc'].append(4)
# >>> a['abc']
# [1, 2, 3, 4]
# >>> a = {'abc':{'abc':[1,2,3]}}
# >>> a['abc']['abc']
# [1, 2, 3]
# >>> a['abc']['abc'].append(4)
# >>> a['abc']['abc']
# [1, 2, 3, 4]



# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# pca.fit(X)
# print pca.explained_variance_ratio_
# print pca.explained_variance_

# 　　　　输出如下：

# [ 0.98318212  0.00850037  0.00831751]
# [ 3.78483785  0.03272285  0.03201892]

# 　　　　可以看出投影后三个特征维度的方差比例大约为98.3%：0.8%：0.8%。投影后第一个特征占了绝大多数的主成分比例。

# 　　　　现在我们来进行降维，从三维降到2维，代码如下：

# pca = PCA(n_components=2)
# pca.fit(X)
# print pca.explained_variance_ratio_
# print pca.explained_variance_

# 　　　　输出如下：

# [ 0.98318212  0.00850037]
# [ 3.78483785  0.03272285]

# 　　　　这个结果其实可以预料，因为上面三个投影后的特征维度的方差分别为：[ 3.78483785  0.03272285  0.03201892]，投影到二维后选择的肯定是前两个特征，而抛弃第三个特征。

# 　　　　为了有个直观的认识，我们看看此时转化后的数据分布，代码如下：

# X_new = pca.transform(X)
# plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
# plt.show()





# pca = PCA(n_components=0.95)
# pca.fit(X)
# print pca.explained_variance_ratio_
# print pca.explained_variance_
# print pca.n_components_

# 我们指定了主成分至少占95%，输出如下：

#[ 0.98318212]
#[ 3.78483785]
#1


