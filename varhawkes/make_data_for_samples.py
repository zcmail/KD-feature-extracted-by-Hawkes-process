import xlrd
import pandas as pd
import numpy as np

#time：2019年12月17日
#author：ZhangChang

#function description：函数描述
#处理CMU数据集
#user 是用户名,line_start和line_end
#fileName是数据文件名

#测试用例
#fileName = "DSL-StrongPasswordData.xls"
#keystroke_data = make_data('s002',0,2,fileName)
#print (keystroke_data)


def make_data(user,line_start,line_end,fileName):
    df_oral = pd.read_excel(fileName)
    df = df_oral[df_oral["subject"]==user]
    make_data = []  # 生成击键行为数据，每次（step）击键数据对应一个实例，一个实例对应于一个二维数组，每个key对应外围，key的down和up对应内围。
    data_instance = []  # 一次击键行为对应的数据。
    data_instance_app = [] #一次击键行为对应数据的归一化处理
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_DD = 0.0  # Down时间
    sum_Hold = 0.0  # Hold 时间
    '''
    以取每个session的前40行作为训练数据，后10行为测试数据为例：
    s002的第1个session的训练数据为line_start=0,line_end=39
    测试数据为line_start=40,line_end=49
    '''
    for i in range(line_start,line_end):
        for j in range(11):
            if j == 0:
                temp_DD = 0.000001  # down时间。
                data_temp.append(temp_DD)  # down事件，start，初始为0，这里用一个很小的值。
            else:
                temp_DD = df.iloc[i, 1 + j * 3]   # down时间。
                sum_DD += temp_DD  # down事件。
                data_temp.append(sum_DD)        # down事件加入list。
            temp_Hole = df.iloc[i, 3 + j * 3]     # Hold的时间。
            #print("temp_DD:")
            #print(temp_DD)
            #print("temp_Hole:")
            #print(temp_Hole)
            sum_Hold = sum_DD + temp_Hole  # up事件。
            data_temp.append(sum_Hold)     # up事件加入list。
            data_instance.append(np.array(data_temp))
            data_temp = []
        ## mean and std 归一化处理
        '''
        mu = np.mean(data_instance)
        sigma = np.std(data_instance)
        data_instance_app = (lambda x: (x - mu) / sigma)(data_instance)
        '''
        make_data.append(data_instance)
        data_instance=[]
        sum_DD = 0.0
    return make_data

def make_data_all(fileName):
    df = pd.read_excel(fileName)
    make_data = []  # 生成击键行为数据，每次（step）击键数据对应一个实例，一个实例对应于一个二维数组，每个key对应外围，key的down和up对应内围。
    data_instance = []  # 一次击键行为对应的数据。
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_DD = 0.0  # Down时间
    sum_Hold = 0.0  # Hold 时间
    '''
    以取每个session的前40行作为训练数据，后10行为测试数据为例：
    s002的第1个session的训练数据为line_start=0,line_end=39
    测试数据为line_start=40,line_end=49
    '''
    for i in range(0,20400):
        for j in range(11):
            if j == 0:
                temp_DD = 0.000001  # down时间。
                data_temp.append(temp_DD)  # down事件，start，初始为0，这里用一个很小的值。
            else:
                temp_DD = df.iloc[i, 1 + j * 3]   # down时间。
                sum_DD += temp_DD  # down事件。
                data_temp.append(sum_DD)        # down事件加入list。
            temp_Hole = df.iloc[i, 3 + j * 3]     # Hold的时间。
            #print("temp_DD:")
            #print(temp_DD)
            #print("temp_Hole:")
            #print(temp_Hole)
            sum_Hold = sum_DD + temp_Hole  # up事件。
            data_temp.append(sum_Hold)     # up事件加入list。
            data_instance.append(np.array(data_temp))
            data_temp = []
        make_data.append(data_instance)
        data_instance=[]
        sum_DD = 0.0
    return make_data


def make_data_all_expect(fileName,user):
    df = pd.read_excel(fileName)
    make_data = []  # 生成击键行为数据，每次（step）击键数据对应一个实例，一个实例对应于一个二维数组，每个key对应外围，key的down和up对应内围。
    data_instance = []  # 一次击键行为对应的数据。
    data_temp = []  # 临时list变量。
    temp_DD = 0.0  # Down-Down临时变量
    temp_Hole = 0.0  # Hold临时编码
    sum_DD = 0.0  # Down时间
    sum_Hold = 0.0  # Hold 时间
    '''
    以取每个session的前40行作为训练数据，后10行为测试数据为例：
    s002的第1个session的训练数据为line_start=0,line_end=39
    测试数据为line_start=40,line_end=49
    '''
    for i in range(0,20400):
        #跳过user的数据
        if df.iloc[i,0] == user:
            #print(df.ix[i,0])
            continue
        #只取每个用户的前50个样本
        if i%400 >= 50:
            continue
        for j in range(11):
            if j == 0:
                temp_DD = 0.000001  # down时间。
                data_temp.append(temp_DD)  # down事件，start，初始为0，这里用一个很小的值。
            else:
                temp_DD = df.iloc[i, 1 + j * 3]   # down时间。
                sum_DD += temp_DD  # down事件。
                data_temp.append(sum_DD)        # down事件加入list。
            temp_Hole = df.iloc[i, 3 + j * 3]     # Hold的时间。
            #print("temp_DD:")
            #print(temp_DD)
            #print("temp_Hole:")
            #print(temp_Hole)
            sum_Hold = sum_DD + temp_Hole  # up事件。
            data_temp.append(sum_Hold)     # up事件加入list。
            data_instance.append(np.array(data_temp))
            data_temp = []
        make_data.append(data_instance)
        data_instance=[]
        sum_DD = 0.0
    return make_data

