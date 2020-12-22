import pandas as pd
import numpy as np

#time：2019年12月17日
#author：ZhangChang

#function description：函数描述
#处理CMU数据集，生产一次事件的数据
#user 是用户名,line_start
#fileName是数据文件名

#测试用例
'''
fileName = "./data/DSL-StrongPasswordData.xls"
keystroke_data,end_time = make_estimate_data('s002',0,fileName)
print (keystroke_data,end_time)
'''

def make_estimate_data(user,line_start,fileName):
    df_oral = pd.read_excel(fileName)
    df = df_oral[df_oral["subject"]==user]
    data_instance = []  # 一次击键行为对应的数据。
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_DD = 0.0  # Down时间
    sum_Hold = 0.0  # Hold 时间
    for j in range(11):
        if j == 0:
            temp_DD = 0.000001  # down时间。
            data_temp.append(temp_DD)  # down事件，start，初始为0，这里用一个很小的值。
        else:
            temp_DD = df.ix[line_start, 1 + j * 3]  # down时间。
            sum_DD += temp_DD  # down事件。
            data_temp.append(sum_DD)  # down事件加入list。
        temp_Hole = df.ix[line_start, 3 + j * 3]  # Hold的时间。
        #print("temp_DD:")
        #print(temp_DD)
        #print("temp_Hole:")
        #print(temp_Hole)
        sum_Hold = sum_DD + temp_Hole  # up事件。
        data_temp.append(sum_Hold)  # up事件加入list。
        data_instance.append(np.array(data_temp))
        data_temp = []
    end_time = max(data_instance[10])
    return data_instance,end_time

'''
fileName = "./data/DSL-StrongPasswordData.xls"
keystroke_data,end_time = make_estimate_data('s002',0,fileName)
print (keystroke_data,end_time)
'''