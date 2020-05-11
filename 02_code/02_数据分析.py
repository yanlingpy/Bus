'''
查看原始数据各个字段包含的值
'''
import datetime
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'

data = pd.read_csv('../01_data/gd_train_data.txt',\
                         header=None,
                         names=['use_city', 'line_name', 'terminal_id', 'card_id', 
                         'create_city', 'deal_time','card_type'])
'''
'use_city'   ：使用地
'line_name'  ：线路名称
'terminal_id'：终端ID
'card_id'    ：卡号
'create_city'：发卡地
'deal_time'  ：交易时间
'card_type'  ：卡类型
'''
print(data.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8926605 entries, 0 to 8926604
Data columns (total 7 columns):
 #   Column       Dtype 
---  ------       ----- 
 0   use_city     object
 1   line_name    int64 
 2   terminal_id  object
 3   card_id      object
 4   create_city  object
 5   deal_time    int64 
 6   card_type    object
dtypes: int64(2), object(5)
memory usage: 476.7+ MB
'''
print(data.describe())#描述性统计
'''
          line_name     deal_time
count  8.926605e+06  8.926605e+06   #计数
mean   3.733057e+02  2.014102e+09   #平均值
std    1.330206e+02  1.392176e+04   #标准差
min    2.810000e+02  2.014080e+09   #最小值
25%    2.810000e+02  2.014091e+09   #较低百分位
50%    2.810000e+02  2.014102e+09   #中位数
75%    5.650000e+02  2.014113e+09   #较高百分位
max    5.650000e+02  2.014123e+09   #最大值
'''
print(data['use_city'].value_counts())
'''
广州    8926605
Name: use_city, dtype: int64
'''
print(data['line_name'].value_counts())
'''
281    6025280
565    2901325
'''
print(data['create_city'].value_counts())
'''
广州     8284418
佛山      552925
汕尾       39094
揭阳       11409
河源       10152
潮州        9151
肇庆        5328
清远        4563
江门        3122
韶关        2222
茂名        1952
岭南通        950
惠州         527
珠海         339
阳江         293
澳门         108
湛江          29
云浮          21
梅州           1
中山           1
'''
print(data['card_type'].drop_duplicates())
'''
0            普通卡
9            老人卡
15           学生卡
30           残疾卡
167          员工卡
5083       治安监督卡
3636568      军属卡
'''
#=========================================================================
#
#数据处理
#日期
data['date'] = data['deal_time'].apply(lambda x:str(x)[:8])
data['time'] = data['deal_time'].apply(lambda x:str(x)[8:])

#小时 与 乘车数量 关系
ax = data['time'].value_counts().sort_index().plot(kind = "bar",title = "小时 与 乘车人数 关系图")
ax.set_ylabel('乘车人数')
ax.set_xlabel('小时')
plt.show()
print("早上 7~10时 乘车人数较多, 下午16~18时 乘车人数较多")

#日乘车分布图
bx=data['date'].value_counts().sort_index().plot(kind = "line",figsize = (20,5),marker='o',color='r', title = "日乘车量")
bx.set_ylabel('乘车人数')
bx.set_xlabel('日期')
plt.show()
print("数据时间分布区间为: %s ~ %s "%(str(min(data['date'])),\
                                   str(max(data['date']))))
print("国庆期间,乘车人数略微下降了")

#===============================

#增加星期信息
data['week'] = data['date'].\
               map(lambda x : \
                   datetime.datetime.strptime(x,'%Y%m%d').strftime("%w"))
cx=data['week'].value_counts().sort_index().plot(kind = "bar",title = "星期 与 乘车人数 关系图")
cx.set_ylabel('乘车人数')
cx.set_xlabel('星期')
print("星期天是 第0天")


print(data.columns)#获取列名
'''
原始字段
'use_city'   ：使用地        string
'line_name'  ：线路名称      int
'terminal_id'：终端ID        string
'card_id'    ：卡号          string
'create_city'：发卡地        string
'deal_time'  ：交易时间      int       例如: 2014082016
'card_type'  ：卡类型        string    例如:'普通卡'

增加字段
date        :交易日期     string    '20140820'
time        :交易小时     string    '16'
week        :星期天       string    '2'
'''

data[(data['time']>='06')&(data['time']<='21')][['deal_time','line_name']].to_csv('../11_data_result/data.csv',index = False)
