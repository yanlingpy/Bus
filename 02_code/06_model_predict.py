#测试，调用模型
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error      
from numpy import concatenate
import matplotlib.pyplot as plt  # 可视化绘制
from sklearn.externals import joblib
import matplotlib

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'
#1 加载模型
model_565 = joblib.load('../03_model/liner_565.model')
scaler_565 = joblib.load('../03_model/scalar_565.model')

model_281 = joblib.load('../03_model/liner_281.model')
scaler_281  = joblib.load('../03_model/scalar_281.model')


#2 读取数据
data_565 = pd.read_csv("../11_data_result/ft_565.csv")
data_565 = data_565.sort_values(by="date_hour" , ascending=True)
data_565.reset_index(drop = True , inplace = True)

data_281 = pd.read_csv("../11_data_result/ft_281.csv")
data_281 = data_281.sort_values(by="date_hour" , ascending=True)
data_281.reset_index(drop = True , inplace = True)


feature_list = list(data_281.columns)
feature_list.remove('label')
feature_list.remove('date_hour')
feature_list.remove('date_day')

#预测
data = data_281
scaler = scaler_281
start_time = 20141205
end_time = 20141212
model = model_281

def predict(data, scaler, model ,start_time, end_time):
    data_one = data[(data['date_day'] >= start_time) & (data['date_day'] <= end_time) ]
    feature_list = list(data_one.columns)
    feature_list.remove('label')
    feature_list.remove('date_hour')
    feature_list.remove('date_day')
    #数据标准化
    X = scaler.fit_transform(data_one[feature_list])
    Y = data_one['label'].values
    #预测
    y_pred = model.predict(X)
    #错误率
    error = 0
    for i in range(len(Y)):
        error = abs(Y[i] - y_pred[i])/Y[i] + error
    error_rate = error/len(Y)
    
    fig = plt.figure()
    ax1 =fig.add_subplot(111)
    ax1.set_title(str(start_time) + "~" + str(end_time) ) #设置标题 
    plt.xlabel('时间')  #设置X轴标签  
    plt.ylabel('乘客数量') #设置Y轴标签  
    plt.plot(list(data_one.index), list(Y),c='r') #画散点图 
    plt.plot(list(data_one.index), list(y_pred),c='b') #画散点图 
    plt.legend(["原始","预测"])
    plt.savefig("..\\11_data_result\\model_pic.jpg", transparent=True, dpi=300, pad_inches = 0)  
    
    return error_rate

predict(data_281,scaler_281,model_281,20141201,20141229)
predict(data_565,scaler_565,model_565,20141225,20141229)
