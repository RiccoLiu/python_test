#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

'''
    []: 列表 list 
    (): 元组 tuple
    {}: 集合 set, 可以用序列初始化 set(序列) 
    frozenset: 只读集合，需要用序列初始化 frozenset(序列)
    {key : val}: 字典 dict
    
        a = 'hello', # 创建元组  ('hello') !! 有逗号就是元组，无逗号就是字符串
        a = ('hello') # 创建的字符串 hello 
        a = {'hello'} # 创建的集合 {'hello'}
        a = set('hello') # 创建的集合 {'e', 'o', 'h', 'l'} !! 这里可以把 'hello' 认为是一个序列，对这个序列取集合后就是一个个的不重复的子母
        a = frozenset('hello') # 创建的不可变集合 {'e', 'o', 'h', 'l'} 
'''

'''
    序列 - 支持切片操作
        列表，元组，字符串
    
    可变对象 - 支持推导式  [expression for item in iterable if condition]
        列表：
            append(x), extend(it), insert(idx, x), remove(x), pop(idx), clear(), reverse(), count(x), index(x)
        集合：
            add, remove, discard(删除元素不存在时不会抛异常)

            合集: a | b
            交集: a & b
            差集: a - b
            对称差集: a ^ b

        字典：
            list, in, del

    不可变元素:
        字符串：
        元组:
        只读集合:

    循环技巧：
        enumerate: 用于取序列的索引和值 (对于字典取的是键)
        zip: 对于多个序列，可以将其内的元素一一匹配
        reversed: 反转 - reversed(range(len(list_test))) 逆向输出
        sorted: 排序
        set: 去重

        del语句:
            删除列表元素通过下标或者切片删除
                del list_test[-2:]

            删除字典元素通过键删除
                if 3 in dict_test:
                    del dict_test[3]
    
    条件语句:
        数值运算：
        比较运算符： in, not in
            支持链式操作: a < b == c 校验 a 是否小于 b, 且 b 是否等于 c。
        
        布尔运算符: and, or, not 取反

        海象运算符：:=
            支持在条件语句中的赋值操作

                # 传统做法                      # 使用海象运算符
                value = get_value()             if value := get_value() is not None:
                if value is not None:    ===>       print(value)
                    print(value)
            
'''

if __name__ == '__main__':
    '''
        deque: 双向队列
            从左端添加: appendleft()
            从右端添加: append()
            从左端删除: popleft()
            从右端删除: pop()
    '''
    
    deque_num_str = deque([1, 2, 3, 'a', 'b', 'c'])
    print('deque_num_str:', deque_num_str)

    deque_num_str.appendleft(0)
    deque_num_str.append('d')
    print('deque_num_str:', deque_num_str)

    deque_num_str.pop()
    deque_num_str.popleft()
    print('deque_num_str:', deque_num_str)

    '''
        列表(字典)推导式：
            [expression for item in iterable if condition]
    '''

    # 列表推导式生成列表
    list_sqr_num = [ x**2  for x in range(9)]
    print('list_sqr_num:', list_sqr_num)

    list_sqr_num = [ x**2  for x in range(9) if x % 2 == 0]
    print('list_sqr_num:', list_sqr_num)

    list_sqr_num = [(x, y) for x in range(5) if x % 2 == 0 for y in range(5) if y %2 == 1]
    print('list_sqr_num:', list_sqr_num)

    # 从列表中筛选出特定条件的元素
    list_str = ['apple', 'bat', 'ball', 'cat', 'banana']
    list_str_filter = [x for x in list_str if len(x) <= 3]
    print('list_str_filter:', list_str_filter)

    # 将列表中的数字转换为字符串
    list_num = [1, 2, 3, 4]
    list_str = [str(x) for x in list_num]
    print(list_str)

    # 从嵌套列表中提取所有元素
    nested_list = [[1, 2], [3, 4], [5, 6]]
    flat_list = [item for sub in nested_list for item in sub]
    print(flat_list)

    # 集合推导式
    letter_set = {x for x in 'abcdefg' if x not in 'aceg' }
    print('letter_set:', letter_set)

    # 字典推导式
    square_dict = {x : str(x) for x in range(5)}
    print('square_dict:', square_dict)

    '''
        元组，序列
            列表 & 字符串 & 元组 等序列支持索引和切片操作, 但元组是不可变对象
    '''

    # 序列输入默认是元组
    t = 12345, 54321, 'hello!'
    print('t:', t)

    # 元组是不可变元素
    # t[0] = 54321

    # 支持索引和切片
    print('t[1:] = ', t[1:], 't[0] = ', t[0])

    # 创建空元组:(), 创建只有一个元素的元组: a = 'hello', 
    a = ()
    print('a:', a)

    a = 'hello',
    print('a:', a)

    # !! 下面这样创建的是字符串,
    a = ('hello')
    print('a:', a, 'type(a):', type(a))

    '''
        集合: 不重复元素组成的无序容器
            支持成员检测、消除重复元素, 支持合集、交集、差集、对称差分等数学运算。

            add(): 添加元素
            remove(): 删除元素
    '''
    
    # !! 这样创建的是空字典
    set_test = {}
    print('type(set_test)', type(set_test))

    set_test = set()
    print('type(set_test)', type(set_test))

    a = set('abcd')
    a.add(3)
    print('a:', a)

    b = set('bdef')
    b.add(5)
    print('b:', b)

    # 合集
    print('a | b: ', a | b)

    # 交集
    print('a & b: ', a & b)

    # 差集
    print('a - b: ', a - b)

    # 对称差分, 只在 a 或者 b 中的元素
    print('a ^ b:', a ^ b)

    fs1 = frozenset([1, 2, 3, 4])
    fs2 = frozenset(['hello'])

    print('fs1:', fs1)
    print('fs2:', fs2)

    '''
        字典: 按照 键:值 存储, 键是唯一的, 插入相同的键时会覆盖旧的键值,
            list: 列出所有键
            in: 是否含有某个键
    '''

    # 直接创建字典
    dict_test = {'jack': 4098, 'sape': 4139}
    print('dict_test:', dict_test)

    # 使用键值对的序列创建字典
    list_pair_test = (['spa', 123], (234, 'smi'), ['pma', 456])
    dict_test = dict(list_pair_test)
    print('dict_test:', dict_test)

    print('list(dict_test):', list(dict_test))
    print('\'jack\' in dict_test:', 'jack' in dict_test)    

    dict_test[123] = 'hello'
    print('dict_test:', dict_test)

    '''
        del语句:
            按照列表索引删除值
            按照字典的键删除键值
            
    '''
    list_test = [1, 2, 3, 4, 5]
    print('list_test:', list_test)

    # 索引删除
    del list_test[0]
    print('list_test:', list_test)

    # 切片删除
    del list_test[-2:]
    print('list_test:', list_test)

    # 清空
    del list_test[:]
    print('list_test:', list_test)

    dict_test = {0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four'}    
    print('dict_test:', dict_test)

    if 3 in dict_test:
        del dict_test[3]

    print('dict_test:', dict_test)

    '''
        循环技巧：
            enumerate: 用于取序列的索引和值 (对于字典取的是键)
            zip: 对于多个序列，可以将其内的元素一一匹配
            reversed: 反转
            sorted: 排序
            set: 去重
    '''

    print('list(list_pair_test):', list(list_pair_test))
    print('list_pair_test:', list_pair_test)

    # items 遍历键值
    for key, val in dict_test.items():
        print(key, ':',val)

    # enumerate 用法
    for idx, val in enumerate(list_pair_test):
        print(idx, ':',val)

    for idx, val in enumerate(dict_test):
        print(idx, ':',val)

    # zip 一一匹配
    print('list_pair_test:', list_pair_test)
    print('dict_test:', dict_test)
 
    for a, b in zip(list_pair_test, dict_test) :
        print('a:', a, ', b:', b)
    
    list_test = [0, 1, 2, 3, 4]
    
    # reversed 反转 range 逆序输出
    for i in reversed(range(len(list_test))):
        print('i:', i, ' val:',list_test[i])

    '''
        序列比较：
            序列对象可以与相同序列类型的其他对象比较
    '''
