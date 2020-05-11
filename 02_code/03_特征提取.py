"""
#提取特征
"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'
#==========================================
#1.读取数据
data = pd.read_csv("../11_data_result/data.csv")
weather_data = pd.read_csv('../11_data_result/weather_data.csv')

data['date'] = data['deal_time'].apply(lambda x:str(x)[:8])
data['time'] = data['deal_time'].apply(lambda x:str(x)[8:])
data_281 = data[data['line_name'] == 281]
data_565 = data[data['line_name'] == 565]

#==========================================
#2.1 数据筛选
#统计每个小时数据量
df_tmp = pd.DataFrame(data_281['deal_time'].value_counts())

#2.2 绘制箱形图：显示一组数据分散情况，不受异常值的影响
df_tmp['deal_time'].plot.box(title="小时客流量（281线路）")
plt.grid(linestyle="--", alpha=0.3)
plt.show()

#2.3 寻找上下限
q3 = df_tmp['deal_time'].quantile(0.75)
q1 = df_tmp['deal_time'].quantile(0.25)
max1 = q3 + 1.5*(q3-q1)
min1 = q1 - 1.5*(q3-q1)
if max1 > df_tmp['deal_time'].max():
    max1 = df_tmp['deal_time'].max()
min2 = q1 - 1.5*(q3-q1)
if min1<df_tmp['deal_time'].min():
    min1 = df_tmp['deal_time'].min()
   
#2.4 删掉异常数据
print("根据常识，6点之前，21点之后小时乘车人数过少，因此判定为异常数据")
save_data = df_tmp[df_tmp['deal_time'] >= min1]
save_data['date'] = save_data.index
save_data = save_data.reset_index(drop = True)

#2.5 星期
save_data['date_week'] = save_data['date'].map(lambda x: \
                        datetime.datetime.strptime(str(x)[:8],'%Y%m%d').strftime("%w"))
#2.6 小时
save_data['date_hour'] = save_data['date'].map(lambda x: str(x)[8:])
save_data['date_day'] =  save_data['date'].map(lambda x: int(str(x)[:8]))

#2.7 独热编码
#2.7.1 小时独特编码
dummy_df = pd.get_dummies(save_data['date_hour'], prefix="date_hour")
save_data = save_data.join(dummy_df)
del save_data['date_hour']

#2.7.2 星期 独热编码
dummy_df = pd.get_dummies(save_data['date_week'], prefix="date_week")
save_data = save_data.join(dummy_df)
del save_data['date_week']

#3 合并天气信息
data_feature = pd.merge(left=save_data, right=weather_data, how='left', left_on='date_day', right_on='day')
print(data_feature.columns)

#3.1 修改列明
data_feature.rename(columns={'deal_time':'label'},inplace=True)
data_feature.rename(columns={'date':'date_hour'},inplace=True)

#=========================================================================
#4.1 增加假期字段
holidays_list = [i + 20141001 for i in range(0,7) ]
data_feature['vacation'] = data_feature['date_day'].map(lambda x: 1 if x in holidays_list else 0 )


pd.DataFrame(data_feature.corr()).to_csv("../11_data_result/ft_281_corr.csv")
data_feature.to_csv("../11_data_result/ft_281.csv",index=False)

#==========================================
#2.1 数据筛选
#统计每个小时数据量
df_tmp = pd.DataFrame(data_565['deal_time'].value_counts())

#2.2 绘制箱形图
df_tmp['deal_time'].plot.box(title="小时客流量（565线路）")
plt.grid(linestyle="--", alpha=0.3)
plt.show()

#2.3 寻找上下限
q3 = df_tmp['deal_time'].quantile(0.75)
q1 = df_tmp['deal_time'].quantile(0.25)
max1 = q3 + 1.5*(q3-q1)
min1 = q1 - 1.5*(q3-q1)
if max1 > df_tmp['deal_time'].max():
    max1 = df_tmp['deal_time'].max()
min2 = q1 - 1.5*(q3-q1)
if min1<df_tmp['deal_time'].min():
    min1 = df_tmp['deal_time'].min()

#2.4 删掉异常数据
print("根据常识，6点之前，21点之后小时乘车人数过少，因此判定为异常数据")
save_data = df_tmp[df_tmp['deal_time'] >= min1]
save_data['date'] = save_data.index
save_data = save_data.reset_index(drop = True)


#2.5 星期
save_data['date_week'] = save_data['date_day'].map(lambda x: \
                        datetime.datetime.strptime(str(x)[:8],'%Y%m%d').strftime("%w"))
#2.6 小时
save_data['date_hour'] = save_data['date'].map(lambda x: str(x)[8:])
save_data['date_day'] =  save_data['date'].map(lambda x: int(str(x)[:8]))

#2.7 独热编码
#2.7.1 小时独特编码
dummy_df = pd.get_dummies(save_data['date_hour'], prefix="date_hour")
save_data = save_data.join(dummy_df)
del save_data['date_hour']

#2.7.2 星期 独热编码
dummy_df = pd.get_dummies(save_data['date_week'], prefix="date_week")
save_data = save_data.join(dummy_df)
del save_data['date_week']

#3 合并天气信息
data_feature = pd.merge(left=save_data, right=weather_data, how='left', left_on='date_day', right_on='day')
print(data_feature.columns)

#3.1 修改列明
data_feature.rename(columns={'deal_time':'label'},inplace=True)
data_feature.rename(columns={'date':'date_hour'},inplace=True)

#=========================================================================
#4.1 增加假期字段
holidays_list = [i + 20141001 for i in range(0,7) ]
data_feature['vacation'] = data_feature['date_day'].map(lambda x: 1 if x in holidays_list else 0 )


pd.DataFrame(data_feature.corr()).to_csv("../11_data_result/ft_565_corr.csv")
data_feature.to_csv("../11_data_result/ft_565.csv",index=False)
