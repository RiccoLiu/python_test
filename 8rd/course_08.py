#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import numpy as np

if __name__ == '__main__':
    '''
        array
    '''
    data_ar = np.array([1, 2, 3, 4])

    print(f'data_ar = {data_ar}, type = {type(data_ar)}')
    print(data_ar)

    '''
        arange: 起始值，终止值， 步长
    '''

    data_ara = np.arange(5)
    print(f'data_ara = {data_ara}, type = {type(data_ara)}')
    
    data_ara = np.arange(3, 8)
    print(f'data_ara = {data_ara}, type = {type(data_ara)}')

    data_ara = np.arange(0, 10, 2)
    print(f'data_ara = {data_ara}, type = {type(data_ara)}')
   
    '''
        random:
            rand(row, col): 
                随机生成row行, col列的数组, 数组中的元素是 [0, 1) 区间内均匀分布的
            
            randn(row, col): 
                随机生成row行, col列的数组, 数组中的元素是 标准正态分布(均值0, 方差1)的随机数

            normal(loc=0.0, scale=1.0, size=None):
                随机生成size行列的数组, 数组中的元素是正态分布的 均值为loc, 标准差为scale 中抽取的随机数
                
            randint(low, high, size): 
                随机生成size行列的数组, 数组中的元素是属于[low, high)范围的随机整数
            
            uniform(low, high, size): 
                随机生成size行列的数组, 数组中的原属是属于[low, high]范围的随机浮点数
            
            choice(arr, size, replace, p): 
                从arr数组中随机抽取size行列的元素, replace决定选择后是否放回去(下次取是否还能取到这个元素), p指取arr每个元素的概率
    
            shuffle:
                打乱原一维数组, 不会返回新的数组

            permutation(x):
                随机排列一维数组 x 的元素，并返回打乱后的新数组。如果 x 是一个整数，则返回一个 [0, x) 的随机排列
                
            seed:
                设置随机生成数的种子，种子值相同，生成的随机数序列也会相同。
                
            binomial:
                
    '''
    arr = np.random.rand(2, 3)
    print(f'arr:\n{arr}, type:{type(arr)}')
    
    arr_n = np.random.randn(2, 3)
    print(f'arr_n:\n{arr_n}, type:{type(arr_n)}')
 
    arr_nor = np.random.normal(0, 1.0, size=(2, 3))
    print(f'arr_nor:\n{arr_nor}, type:{type(arr_nor)}')  

    arr_int = np.random.randint(5, 10, size = (2, 3))
    print(f'arr_int:\n{arr_int}, type:{type(arr_int)}')
 
    arr_float = np.random.uniform(5, 10, size = (2, 3))
    print(f'arr_float:\n{arr_float}, type:{type(arr_float)}')
 
    arr_cho = np.random.choice([1, 2, 3, 4, 5], size=(2, 3), replace=True,
                        p=[0.1, 0.2, 0.3, 0.2, 0.2])
    print(f'arr_cho:\n{arr_cho}, type: {type(arr_cho)}')
    
    test_arr = [1, 2, 3, 4, 5]
    arr_shu = np.random.shuffle(test_arr)
    
    print(f'test_arr:\n{test_arr}, type: {type(test_arr)}')

    test_arr = [1, 2, 3, 4, 5]
    arr_per = np.random.permutation(test_arr)

    print(f'test_arr:\n{test_arr}, type: {type(test_arr)}')
    print(f'arr_per:\n{arr_per}, type: {type(arr_per)}')

    val = 40

    np.random.seed(val)
    arr_seed = np.random.rand(2, 3)
    print(f'arr_seed:\n{arr_seed}, type: {type(arr_seed)}')
     
    np.random.seed(val)
    arr_seed2 = np.random.rand(2, 3)
    print(f'arr_seed2:\n{arr_seed2}, type: {type(arr_seed2)}')

    '''
        矩阵：
            零矩阵：
                mat33_zero = np.zeros((3, 3), dtype=float)

            一矩阵：
                mat33_one = np.ones((3, 3), dtype=int)
                
            单位矩阵:
                mat33_eye = np.eye(3, dtype=int)
            
            随机矩阵：
                mat34_empty = np.empty((3, 4), dtype=object)
    '''
    mat34_zero = np.zeros((3, 4), dtype=float)
    print(f'mat34_zero:\n {mat34_zero} type: {type(mat34_zero)}')

    mat34_one = np.ones((3, 4), dtype=int)
    print(f'mat33_one:\n {mat34_one} type: {type(mat34_one)}')    

    print(f'mat34_one.ndim = {mat34_one.ndim}')#维数
    print(f'mat34_one.size = {mat34_one.size}')#元素个数
    print(f'mat34_one.dtype = {mat34_one.dtype}')#数据类型
    print(f'mat34_one.shape = {mat34_one.shape}')#数组的形状

    mat33_eye = np.eye(3, dtype=int)
    print(f'mat33_eye:\n {mat33_eye} type: {type(mat33_eye)}')    

    mat34_empty = np.empty((3, 4), dtype=object)
    print(f'mat34_empty:\n {mat34_empty} type: {type(mat34_empty)}')    

    mat34_empty[0, 0] = 5
    mat34_empty[1, :] = [1, 2, 3, 4]  # 给第2行赋值
    mat34_empty[:, 2] = [4, 5, 6]  # 给第3列赋值
    
    print(f'mat34_empty:\n {mat34_empty} type: {type(mat34_empty)}')    

    print(f'mat34_empty.ndim = {mat34_empty.ndim}')     #维数
    print(f'mat34_empty.size = {mat34_empty.size}')     #元素个数
    print(f'mat34_empty.dtype = {mat34_empty.dtype}')   #数据类型
    print(f'mat34_empty.shape = {mat34_empty.shape}')   #数组的形状

    '''
        矩阵运算：
            transpose:
                转置
    '''

    '''
        reshape(dim, row, col):
            不改变原数组顺序的情况下返回一个新的dim 维, 每一个维度是 row行 col 列的数组对象，当数量不足时会报错

        resize(dim, row, col):
            不改变原数组顺序的情况下将当前数组修改为dim 维 row行 col 列的数组对象,数量不足时会进行裁减或者补充0
            
        ravel:
            将多维变成一维
    '''
    l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
    arr = np.array(l)

    print(f'arr:\n {arr} type:{type(arr)}, dim: {arr.ndim}')

    mat26 = arr.reshape(2, 6)
    print(f'mat26:\n {mat26} type: {type(mat26)}, dim: {mat26.ndim}')    

    mat232 = mat26.reshape(2, 3, 2)
    print(f'mat232:\n {mat232} type: {type(mat232)}, dim: {mat232.ndim}')    

    l2 = [i+1 for i in range(24)]
    arr2 = np.array(l2)

    mat234 = arr2.reshape(2, 3, 4)
    print(f'mat234:\n {mat234} type: {type(mat234)}, dim: {mat234.ndim}')    
    
    print(f'mat26.reshape(2,-1):\n {mat26.reshape(2,-1)}') # 变成 2行6列
    print(f'mat26.reshape(2,2,-1):\n {mat26.reshape(2,2,-1)}') # 变成 2页2行3列
    print(f'mat26.reshape(1,-1):\n {mat26.reshape(1,-1)}') #变成 1行的(由多维变为1维)
    print(f'mat26.reshape(-1):\n {mat26.reshape(-1)}') #变成 1 维     

    mat26.resize((2, 6))
    print(f'mat26.resize(2, 6):\n {mat26}') #变成 2x3 维 
    
    
    arr = np.array([[1096, 718],
                    [1094, 720],
                    [1094, 724],
                    [1092, 726]], dtype=np.int32)

    print('------------ arr: -------------------------')
    print(arr)
   
    print('------------ arr reshape: -------------------------') 
    arr.reshape((-1, 1, 2))
    print(arr)
    
    mask = np.array([[0, 1, 2],
                    [0, 0, 0]])
    threshold = 0.5

    whe = np.where(mask > threshold)
    print(f'whe:{whe}')
    
    rcs = np.column_stack(whe)
    print(f'rcs:{rcs}')
    
    xys = rcs[:, ::-1]
    print(f'xys:{xys}')