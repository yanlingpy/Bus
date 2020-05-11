import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from linearRegression import linearRegression
import joblib

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'

#拟合曲线
def show_pid(x_values, inv_y ,inv_yhat,title):
    fig = plt.figure()
    ax1 =fig.add_subplot(111)
    ax1.set_title("训练集") #设置标题 
    plt.xlabel('时间')  #设置X轴标签  
    plt.ylabel('乘客数量') #设置Y轴标签  
    plt.plot(list(x_values), list(inv_y),c='r') #画散点图 
    plt.plot(list(x_values), list(inv_yhat),c='b') #画散点图 
    plt.legend(["原始","预测"])
    plt.show()
      
#=============================================================================
#1. 读取数据
data = pd.read_csv("../11_data_result/ft_565.csv")
data = data.sort_values(by="date_hour" , ascending=True)#依据小时升序排列
data.reset_index(drop = True , inplace = True)#把原来的索引index列去掉百，重置index，直接在原数组上对数据进行修改
#2
#2.1 获取特征列
print(data.info())
feature_list = list(data.columns)
feature_list.remove('label')
feature_list.remove('date_hour')
feature_list.remove('date_day')
#星期特征、小时特征、温度特征
'''
['date_hour_06',
 'date_hour_07',
 'date_hour_08',
 'date_hour_09',
 'date_hour_10',
 'date_hour_11',
 'date_hour_12',
 'date_hour_13',
 'date_hour_14',
 'date_hour_15',
 'date_hour_16',
 'date_hour_17',
 'date_hour_18',
 'date_hour_19',
 'date_hour_20',
 'date_hour_21',
 'date_week_0',
 'date_week_1',
 'date_week_2',
 'date_week_3',
 'date_week_4',
 'date_week_5',
 'date_week_6',
 'day',
 'weather_d',
 'weather_n',
 'weather_abs',
 'temperature_h',
 'temperature_l',
 'temperature_average',
 'wind_direction_force_d',
 'wind_direction_force_n',
 'wind_average',
 'wind_abs']
'''

#2.2 划分训练集和测试集
data_565_train = data[data['date_hour'] <  2014122500]
data_565_test  = data[data['date_hour'] >= 2014122500]

#2.3 数据规范化化：z-score规范化
scaler = preprocessing.StandardScaler().fit(data[feature_list])

train_X = scaler.transform(data_565_train[feature_list])
train_Y = data_565_train['label'].values

#创建模型
linear_model = linearRegression(train_X, 0.003, 5000)
#训练
linear_model.fit(train_X, train_Y)

#===========================================================================
#3 训练集
#预测,返回预测标签
result_list =linear_model.predict(train_X)

#计算错误率
error = 0
for i in range(len(train_Y)):
    error = abs(train_Y[i] - result_list[i])/train_Y[i] + error
error_rate_value = error/len(train_Y)

# 计算 RMSE
rmse = sqrt(mean_squared_error(train_Y, result_list))

print("train 错误率:%f"%error_rate_value)
print('Train RMSE: %.3f' % rmse)

#画出拟合曲线
show_pid(data_565_train.index, train_Y ,result_list, "训练集")

#===========================================================================
#测试集
data_565_test = data_565_test.sort_values(by="date_hour" , ascending=True)
data_565_test.reset_index(drop = True, inplace = True)

test_X = scaler.transform(data_565_test[feature_list])
test_Y = data_565_test['label'].values

result_list = linear_model.predict(test_X)

#计算错误率
error = 0
for i in range(len(test_Y)):
    error = abs(test_Y[i] - result_list[i])/test_Y[i] + error
error_rate = error/len(test_Y)

# 计算 RMSE
rmse = sqrt(mean_squared_error(test_Y, result_list))

print("test 错误率:%f"%error_rate)
print('Test RMSE: %.3f' % rmse)

#画出拟合曲线
show_pid(data_565_test.index, test_Y ,result_list, "测试集")


#保存模型
joblib.dump(linear_model, '../03_model/liner_565.model')#多元线性回归模型
joblib.dump(scaler, '../03_model/scalar_565.model')#数据标准化模型
