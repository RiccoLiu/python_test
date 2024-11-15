#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 数据类型转换
# int             整型
# float           浮点型
# complex         复数

# chr             字符
# str             字符串

# ord(x)          字符转整型
# hex(x)          整型转16进制字符串
# oct(x)          整型转8进制字符串
# repr            表达式字符串

# eval(str)       用来计算在字符串中的有效Python表达式,并返回一个对象

# tuple()         元组
# list()          将序列 s 转换为一个列表
# set()           转换为可变集合
# frozenset(s)    不可变集合
# dict(d)         字典


# 数字运算 & 字符串 & 列表 
#
# 一、数字运算 
# + 加法 
# - 减法 
# * 乘法 
# / 除法 - 返回浮点数
# % 取余 
# // 除法 - 向下取整
# ** 乘方 

# 1、混合类型运算时会把整数转换为浮点数
# 2、交互模式下会把上次输出的表达式会赋给变量 _ ,下次计算是可以直接使用 _ 变量
# 3、 ** 比 - 的优先级更高, 所以 -3**2 会被解释成 -(3**2) ，因此，结果是 -9。要避免这个问题，并且得到 9, 可以用 (-3)**2

# 二、字符串
# 使用单引号 ('...') 和双引号 ("...") 表示字符串，反斜杠 (\) 用于转义字符

# 1、如果不想使用转义字符 (\), 可以在字符串前加 (r)
#    print(r'C:\some\name')
#
# 2、字符串可以包含多行时，可以使用 """...""" 或 '''...'''
#
# 3、字符串可以用 + 合并（粘到一起），也可以用 * 重复
#   print(3 * 'un' + 'ium');  print('Py' 'thon')
# 
# 4、字符串拼接
# 4.1.多字符串自动拼接
#    text = ('Put several strings within parentheses '
#           'to have them joined together.')
# 4.2.变量与字符串拼接使用 + 
#
# 5、字符串支持下标访问, 下标可以是负数
#   word = 'Python'; word[0] = 'P';  word[-1] = 'n'
#
# 6、字符串切片 - 下标从 0 开始，不包括末尾最后一个元素
#   word = 'Python'; word[:2] = 'Py';  word[2:] = 'thon'; word[2:6] = 'thon'
#
# 7、下标索引越界会报错，切片越界会自行处理，不会报错
#   word[4:42] = "on"; word[42:] = "";
#
# 8、字符串是不能修改的，如果要想修改应该新建一个新的字符串
#   "j" + word[1:] = "jython";
#
# 9、len() 函数可以返回字符串的长度
#
# 三、列表
# 列表使用方括号标注，逗号分隔的一组值
#   squares = [1, 4, 9, 16, 25]
# 
# 1、列表支持索引和切片, 切片操作返回包含请求元素的新列表 - 深拷贝
#   squares[-1]; squares[-3:];
#
# 2、列表支持合并操作
#   squares + [36, 49, 64, 81, 100]
#
# 3、列表的元素是可以通过下标修改的
#   squares[0] = 1234; 
#
# 4、append() 函数可以在列表末尾添加新的元素
# 
# 5、列表为切片赋值可以改变列表大小，甚至清空整个列表：
# >>> letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# >>> letters
# ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# >>> # replace some values
# >>> letters[2:5] = ['C', 'D', 'E']
# >>> letters
# ['a', 'b', 'C', 'D', 'E', 'f', 'g']
# >>> # now remove them
# >>> letters[2:5] = []
# >>> letters
# ['a', 'b', 'f', 'g']
# >>> # clear the list by replacing all the elements with an empty list
# >>> letters[:] = []
# >>> letters
#  
# 6、len() 函数也可以返回列表的长度
#
# 7、列表可以嵌套列表 - 包含其他列表的列表
# >>> a = ['a', 'b', 'c']
# >>> n = [1, 2, 3]
# >>> x = [a, n]
# >>> x
# [['a', 'b', 'c'], [1, 2, 3]]
# >>> x[0]
# ['a', 'b', 'c']
# >>> x[0][1]
# 'b'

if __name__ == '__main__':
    # Fibonacci series:
    # the sum of two elements defines the next
    a, b = 0, 1
    while (a < 1000) :
        a, b = b, a+b
        if b < 1000 :
            print(a, end = ",")
        else :
            print(a)
            break

# 1、多重赋值，a, b = 0, 1, a, b = b, a+b

# 2、print() 函数输出给定参数的值。除了可以以单一的表达式作为参数（比如，前面的计算器的例子），它还能处理多个参数，包括浮点数与字符串。
# 它输出的字符串不带引号，且各参数项之间会插入一个空格，这样可以实现更好的格式化操作：


