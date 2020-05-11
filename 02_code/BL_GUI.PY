import tkinter as tk
from PIL import Image,ImageTk
from sklearn.externals import joblib
import pandas as pd
import matplotlib.pyplot as plt  # 可视化绘制
import matplotlib
from sklearn.metrics import mean_squared_error      
from math import sqrt

myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['font.family']='sans-serif'

#1 加载模型
model_565 = joblib.load('../03_model/liner_565.model')#多元线性回归模型
scaler_565 = joblib.load('../03_model/scalar_565.model')#数据标准化模型

model_281 = joblib.load('../03_model/liner_281.model')
scaler_281  = joblib.load('../03_model/scalar_281.model')


#2 读取数据
data_565 = pd.read_csv("../11_data_result/ft_565.csv")
data_565 = data_565.sort_values(by="date_hour" , ascending=True)#依据小时升序排列
data_565.reset_index(drop = True , inplace = True)#把原来的索引index列去掉百，重置index，直接在原数组上对数据进行修改

data_281 = pd.read_csv("../11_data_result/ft_281.csv")
data_281 = data_281.sort_values(by="date_hour" , ascending=True)
data_281.reset_index(drop = True , inplace = True)


def predict(data, scaler, model ,start_time, end_time,name):
    data_one = data[(data['date_day'] >= start_time) & (data['date_day'] <= end_time) ]
    feature_list = list(data_one.columns)#修改列名
    feature_list.remove('label')
    feature_list.remove('date_hour')
    feature_list.remove('date_day')
    #数据标准化
    X = scaler.fit_transform(data_one[feature_list])
    Y = data_one['label'].values#返回值
    #预测
    y_pred = model.predict(X)
    #错误率、RMSE
    error = 0
    for i in range(len(Y)):
        error = abs(Y[i] - y_pred[i])/Y[i] + error
    error_rate = error/len(Y)
    rmse = sqrt(mean_squared_error(y_pred, Y))
    
    fig = plt.figure()
    ax1 =fig.add_subplot(111)
    ax1.set_title(str(start_time) + "~" + str(end_time) ) #设置标题 
    plt.xlabel('时间')  #设置X轴标签  
    plt.ylabel('乘客数量') #设置Y轴标签  
    plt.plot(list(data_one.index), list(Y),c='r') #画散点图 ：原始
    plt.plot(list(data_one.index), list(y_pred),c='b') #画散点图 ：预测
    plt.legend(["原始","预测"])#图例
    plt.savefig("..\\11_data_result\\model_pic_%s.jpg"%str(name), transparent=True, dpi=300, pad_inches = 0)  
    return error_rate,rmse
  
window=tk.Tk()
window.title("客流量预测软件")
window.geometry("800x600")
#==============================================================================

def show_f_pid_one_565():
    global image1,im1
    global line_num
    error_rate,rmse = predict(data_565,scaler_565,model_565,int(e_start_565.get()),int(e_end_565.get()),565)
    print(error_rate)
    image1=Image.open("../11_data_result/model_pic_565.jpg")
    w,h=image1.size
    print(w)
    print(h)
    image_resized=resize(340,450,image1)
    im1 = ImageTk.PhotoImage(image_resized)
    canvas_565.create_image(180,130, anchor='center', image=im1)
    t_565.delete(1.0,tk.END)
    t_565.insert('insert', "错误率:%s\n"%(str(round(error_rate,2))))
    t_565.insert('insert', "RMSE:%s\n"%(str(round(rmse,2))))
    
def show_f_pid_one_281():
    global image,im2
    global line_num
    error_rate,rmse = predict(data_281,scaler_281,model_281,int(e_start.get()),int(e_end.get()),281)
    print(error_rate)
    image=Image.open("../11_data_result/model_pic_281.jpg")
    w,h=image.size
    print(w)
    print(h)
    image_resized=resize(340,450,image)
    im2 = ImageTk.PhotoImage(image_resized)
    canvas_281.create_image(180,130, anchor='center', image=im2)
    t_281.delete(1.0,tk.END)
    t_281.insert('insert', "错误率:%s\n"%(str(round(error_rate,2))))
    t_281.insert('insert', "RMSE:%s\n"%(str(round(rmse,2))))
#==============================================================================

def resize(w_box, h_box, pil_image) :  # 参数是：要适应的窗口宽、高、Image.open后的图片
    w, h = pil_image.size  # 获取图像的原始大小
    f1 = 1.0 * w_box / w
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)
#==============================================================================
#设置
l = tk.Label(window, \
             text='281线路', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l.place(x=10, y=10, anchor='nw')

l_start = tk.Label(window, \
             text='起始时间:', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l_start.place(x=10, y=60, anchor='nw')

e_start = tk.Entry(window, show = None, font=('Arial', 14),width=10)#显示成明文形式
e_start.place(x=100,y=65)

l_end = tk.Label(window, \
             text='结束时间:', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l_end.place(x=10, y=100, anchor='nw')

e_end = tk.Entry(window, show = None, font=('Arial', 14),width=10)#显示成明文形式
e_end.place(x=100,y=105)

b_281 = tk.Button(window, \
              text='预测', \
              font=('Arial', 12), \
              width=12, \
              height=1,\
              command= show_f_pid_one_281 )
b_281.place(x=20, y=150, anchor='nw')

canvas_281 = tk.Canvas(window, bg='grey', height=280, width=360)
canvas_281.place(x=260,y=10)

t_281 = tk.Text(window, height=21,width=20)
t_281.place(x=640,y=10)
#==============================================================================

l_565 = tk.Label(window, \
             text='565线路', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l_565.place(x=10, y=300, anchor='nw')

l_start_565 = tk.Label(window, \
             text='起始时间:', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l_start_565.place(x=10, y=360, anchor='nw')

e_start_565 = tk.Entry(window, show = None, font=('Arial', 14),width=10)#显示成明文形式
e_start_565.place(x=100,y=365)

l_end_565 = tk.Label(window, \
             text='结束时间:', \
             font=('Arial', 12), \
             width=10, \
             height=2)
l_end_565.place(x=10, y=400, anchor='nw')

e_end_565 = tk.Entry(window, show = None, font=('Arial', 14),width=10)#显示成明文形式
e_end_565.place(x=100,y=405)

b_565 = tk.Button(window, \
              text='预测', \
              font=('Arial', 12), \
              width=12, \
              height=1,\
              command= show_f_pid_one_565 )
b_565.place(x=20, y=450, anchor='nw')

canvas_565 = tk.Canvas(window, bg='grey', height=280, width=360)
canvas_565.place(x=260,y=310)

t_565 = tk.Text(window, height=21,width=20)
t_565.place(x=640,y=310)

window.mainloop()
# 注意，loop因为是循环的意思，window.mainloop就会让window不断的刷新，如果没有mainloop,就是一个静态的window,传入进去的值就不会有循环，mainloop就相当于一个很大的while循环，有个while，每点击一次就会更新一次，所以我们必须要有循环
# 所有的窗口文件都必须有类似的mainloop函数，mainloop是窗口文件的关键的关键。
