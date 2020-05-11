import re
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')#设置字体：宋体
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
matplotlib.rcParams['font.family']='sans-serif'    #用于显示字体的名字

#读取原始数据
weather_report = pd.read_csv('../01_data/gd_weather_report.txt',\
                             header=None,\
                             names=['date', 'weather', 'temperature', \
                                    'wind_direction_force'])
#数据类型装换
#2015/1/27 => 20150127
def changeDate(date):
    dateList = date.split('/')
    if int(dateList[1]) < 10:
        month = '0' + dateList[1]
    else:
        month = dateList[1]
    if int(dateList[2]) < 10:
        day = '0' + dateList[2]
    else:
        day = dateList[2]
    return int(dateList[0] + month + day)

#1. 打印数据表信息
print("原始数据包括：日期、温度、天气状况、风力风向")
print(weather_report.info())
'''
data                    日期
weather                 天气状况
temperature             温度
wind_direction_force    风向风力
'''
#==============================================================================
#1.1 
#时间字符串转换
weather_report['day'] = weather_report['date'].apply(lambda x: changeDate(x))
weather_report.set_index(["date"], inplace=True)
print("时间分布: %s ~ %s"%(min(weather_report['day']),
                         max(weather_report['day'])))
#==============================================================================
#1.2 
#天气字段处理
#1.2.1 天气（白天）
weather_report['weather_d'] = weather_report['weather'].apply(lambda x: x.split('/')[0])
a=weather_report['weather_d'].value_counts().plot(kind='bar',title = "天气状况柱状图（白天）")
a.set_ylabel('次数')
a.set_xlabel('天气状况（白天）')
plt.show()

#1.2.2 天气（夜间）
weather_report['weather_n'] = weather_report['weather'].apply(lambda x: x.split('/')[1])
b=weather_report['weather_n'].value_counts().plot(kind='bar',title = "天气状况柱状图（夜间）")
b.set_ylabel('次数')
b.set_xlabel('天气状况（夜间）')
plt.show()

weathermap = {'多云': 0, \
              '晴': 0, \
              '阴': 0, \
              '霾': 0, \
              '雷阵雨': 1, \
              '小雨': 2, \
              '小到中雨': 2, \
              '中雨': 2, \
              '中到大雨': 2, \
              '大雨': 2}

#天气（白天）、天气（夜间）  编码
weather_report['weather_d'] = weather_report['weather_d'].map(weathermap)
weather_report['weather_n'] = weather_report['weather_n'].map(weathermap)
weather_report['weather_abs'] = abs(weather_report['weather_d'] - weather_report['weather_n'])

#同一天天气转变情况
c=weather_report['weather_abs'].value_counts().plot(kind='bar',title = "单日天气变化柱状图")
c.set_ylabel('次数')
c.set_xlabel('天气状况（差值）')
plt.show()

#===========================================================================

#1.3 
#温度字符串处理
#1.3.1
#最高温度
weather_report['temperature_h'] = weather_report['temperature'].apply(lambda x: int(re.sub(r'\D', '', x.split('/')[0])))
d=weather_report['temperature_h'].value_counts().sort_index().plot(kind='line',marker='o',color='r',title = "单日最高温度分布图")
d.set_ylabel('次数')
d.set_xlabel('最高温度')
plt.show()
#最低温度
weather_report['temperature_l'] = weather_report['temperature'].apply(lambda x: int(re.sub(r'\D', '', x.split('/')[1])))
e=weather_report['temperature_l'].value_counts().sort_index().plot(title = "单日最低温度分布图")
e.set_ylabel('次数')
e.set_xlabel('最低温度')
plt.show()

#每日温度变化
weather_report['temperature_h'].plot(kind='line',marker='o',color='r',figsize = (20,5), title = "单日温度分布")
weather_report['temperature_l'].plot(kind='line',marker='o',color='b',figsize = (20,5))
plt.show()

#平均温度
weather_report['temperature_average'] = (weather_report['temperature_h'] + weather_report['temperature_l']) / 2.0
f=weather_report['temperature_average'].value_counts().sort_index().plot(kind='bar',title = "日平均温度分布图")
f.set_ylabel('次数')
f.set_xlabel('平均温度')
plt.show()

#===========================================================================
#1.4 
#风速风向（白天）
weather_report['wind_direction_force_d'] = weather_report['wind_direction_force'].apply(lambda x: x.split('/')[0])
g=weather_report['wind_direction_force_d'].value_counts().plot(kind='bar',title = "风力(白天）柱状图")
g.set_ylabel('次数')
g.set_xlabel('风力（白天）')
plt.show()
#风速风向（夜间）
weather_report['wind_direction_force_n'] = weather_report['wind_direction_force'].apply(lambda x: x.split('/')[1])
h=weather_report['wind_direction_force_n'].value_counts().plot(kind='bar',title = "风力（夜间）柱状图")
h.set_ylabel('次数')
h.set_xlabel('风力（夜间）')
plt.show()
#风向风速 编码
windmap = {'无持续风向≤3级': 0, \
           '无持续风向微风转3-4级': 1, \
           '北风微风转3-4级': 1, \
           '东北风3-4级': 1, \
           '北风3-4级': 1,\
           '东南风3-4级': 1,\
           '东风4-5级': 2,\
           '北风4-5级': 2}
#风向风速1
weather_report['wind_direction_force_d'] = weather_report['wind_direction_force_d'].map(windmap)
#风向风速2
weather_report['wind_direction_force_n'] = weather_report['wind_direction_force_n'].map(windmap)
#风向风速 均值
weather_report['wind_average'] = (weather_report['wind_direction_force_d'] + 
                                  weather_report['wind_direction_force_n']) / 2.0
#风向风速 差值
weather_report['wind_abs'] = abs(weather_report['wind_direction_force_n'] - 
                                 weather_report['wind_direction_force_d'])

i=weather_report['wind_average'].value_counts().plot(kind='bar',title = "平均风力柱状图")
i.set_ylabel('次数')
i.set_xlabel('平均风力')
plt.show()
j=weather_report['wind_abs'].value_counts().plot(kind='bar',title = "风力变化柱状图")
j.set_ylabel('次数')
j.set_xlabel('风力变化')
plt.show()

print(weather_report.info())
'''
'weather'                     string     晴/雷阵雨
'temperature'                 string     36℃/26℃
'wind_direction_force'        string     无持续风向≤3级/无持续风向≤3级
'day'                         int     时间 “20140801”
'weather_d',                  int        天气1 编码
'weather_n',                  int        天气2 编码
'weather_abs',                float      |天气2 - 天气1| 绝对值
'temperature_h',              int        最高温度
'temperature_l',              int        最低温度
'temperature_average',        float      温度均值
'wind_direction_force_d',     string     风向1 编码
'wind_direction_force_n',     string     风向2 编码 
'wind_average',               int        风向编码后均值
'wind_abs',                   int        风向编码后差值
'''
feature_list = ['day', 'weather_d', 'weather_n','weather_abs',\
                'temperature_h', 'temperature_l', 'temperature_average',\
                'wind_direction_force_d','wind_direction_force_n', \
                'wind_average','wind_abs']

weather_report_result = weather_report[feature_list]
# 生成标准时间格式的天气数据
weather_report_result = weather_report_result.reset_index(drop=True)
print(weather_report_result)

#填充空值
for column in list(weather_report_result.columns[weather_report_result.isnull().sum() > 0]):
    mean_val = weather_report_result[column].mean()
    weather_report_result[column].fillna(mean_val, inplace=True)

print(np.isnan(weather_report_result).any())
    
weather_report_result.to_csv('../11_data_result/weather_data.csv',index = False)
