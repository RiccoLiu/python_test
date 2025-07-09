#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1、多重赋值，a, b = 0, 1, a, b = b, a+b
# 2、print() 函数输出给定参数的值。除了可以以单一的表达式作为参数（比如，前面的计算器的例子），它还能处理多个参数，包括浮点数与字符串。
# 它输出的字符串不带引号，且各参数项之间会插入一个空格，这样可以实现更好的格式化操作：

if __name__ == '__main__':
    '''
    0、多重赋值, 打印
    '''
    a, b, c = 3, 'five', [3, 5, 'test_strs']
    print('a =', a, 'b =', b, 'c =', c)
    print(f'a = {a}, b = {b}, c = {c}')
    print('a = %d, b = %s, c = %s' % (a, b, c))

    print('-----------------------')

    '''
    1、数字运算
        1、混合类型运算时会把整数转换为浮点数
        2、交互模式下会把上次输出的表达式会赋给变量 _ ,下次计算是可以直接使用 _ 变量
        3、 ** 比 - 的优先级更高, 所以 -3**2 会被解释成 -(3**2) ，因此, 结果是 -9。要避免这个问题,并且得到 9, 可以用 (-3)**2
    '''
    print("15+6 =", 15+6)
    print("15-6 =", 15-6)
    print("15*6 =", 15*6)
    print("15/6 =", 15/6)
    print("15%6 =", 15%6)
    print("15//6 =", 15//6)
    print("15**6 =", 15**6)
    print("(-15)**6 =", (-15)**6)

    print('-----------------------')

    '''
    2、类型转换
        int, float, complex : 整型，浮点型，复数
        chr, str, ord: 字符，字符串, 字符转 unicode 码
        hex, oct: 整型转16进制, 转8进制
        repr, eval: 对象转字符串，执行字符串的表达式
        list, tuple, dict, set, frozenset: 数据结构 列表，元组，字典，集合，不可变集合
    '''
    print('int(\'9754\') = %d' % int('9754'))   # 字符串转整型
    print('str(9754) = %s' % str(9754))         # 整型转字符串

    print('****')

    print('A = %c' % 'A')       # 打印字符
    print('ABC = %s' % 'ABC')   # 打印字符串
    print('ord(\'A\'):', ord('A'), ', ord(\'a\'):', ord('a')) # 打印字符 unicode 码

    print('****')

    print('hex(3913):', hex(3913))  # 转 16进制
    print('oct(3913):', oct(3913))  # 转 8 进制
    print('int(hex(3913), 16):', int(hex(3913), 16)) # 转 10 进制

    print('****')

    list_str = repr([9, 5, 'test', 26]) # 列表转字符串
    print('list_str = %s, len(list_str) = %d, type(list_str) = %s' % (list_str, len(list_str), type(list_str)))

    act_list = eval(list_str)           # 字符串转列表
    print('eval(list_str) = %s, len(ac_list) = %d, type(act_list) = %s' % (act_list, len(act_list), type(act_list)))

    print('-----------------------')

    '''
    3、字符串 - 使用单引号或者双引号表示
        1、字符串包含多行时可以使用三重引号
        2、字符串是只读的, 不能修改
        3、字符串遍历
            字符串可以通过下标访问，下标可以是负数，下标可以用len 配合range函数
            字符串切片 word[:2], 下标从0开始，不包含结束下标的元素
                通过下标访问越界会报错，但切片下标越界不会报错，会自行处理
        4、字符串拼接
            使用 + 用于多字符串的合并，或者字符串和变量的合并
            * 可以用于生成重复的多个字符串
    '''
    print(r'C:\some\name') # 打印反协杠

    print('****')

    # 三重引号
    print('''Usage: thingy [OPTIONS]
    -h                        Display this usage message
    -H hostname               Hostname to connect to ''')

    print("****")

    word = 'hello world '

    # !! 字符串只读，不能像下面这样修改
    # word[6:] = 'lc'
    print(' word[:6] + \'lc\' = ', word[:6] + 'lc')

    print('****')

    # 通过下标遍历字符串
    for i in range(len(word)):
        print(word[i], end = '')
    print()

    # 下标越界报错
    # print('word[12] = ', word[12])

    # 切片越界不会报错
    print('word[12:] = ', word[12:])

    print('****')

    # 字符串拼接
    print('hel' 'lo' ' ' 'wor' 'ld')
    print('word + str(3):', word + str(3))
    print('3 * word = ', 3 * word)

    print('-----------------------')

    '''
    4、列表
        列表使用方括号标注，逗号分隔的一组值，且列表可以嵌套列表

        列表遍历
            可以通过下标遍历，可以切片, 可以通过下标修改，可以通过切片修改
        列表拼接
            + 操作符可以合并两个列表
            list.append() 添加元素到列表末尾
            切片修改列表，切片清空，切片删除等

        !!! 注意：
        可变对象（如列表、字典）：当你将一个可变对象（如列表）赋值给另一个变量时，两个变量会指向同一个对象。如果你通过其中一个变量修改该对象的内容，另一个变量也会受到影响。
        不可变对象（如整数、字符串、元组）：如果你将不可变对象赋值给另一个变量，两个变量会分别持有独立的副本，修改一个变量的值不会影响另一个变量。
    '''

    list_test = [1, 4, 9, 16, 'test', ['sum', 'te', 3, 10]]
    print('list_test:', list_test)

    list_back = list_test[-1]       # 可变对象，这样相当于引用
    # list_back2 = list_test[-1].copy() # 可变对象，相当于引用的浅拷贝，里面的可变元素相当于引用，不可变元素相当于深拷贝, list_test[-1].copy() 等价与 list_test[-1][:]
    list_back[1] = "tim"

    print('list_back:', list_back)
    print('list_test:', list_test)

    # 下标遍历
    for i in range(len(list_test)) :
        print('list_test[i] = ', list_test[i])

    # 列表拼接
    print('list_test + [9, 8, 7] = ', list_test + [9, 8, 7])

    list_test.append(3)
    print('list_test.append(3):', list_test)

    # 切片修改
    list_test[:3] = [34, 28]
    print('list_test:', list_test)

    # 切片删除指定下标的元素
    list_test = list_test[:3]
    print('list_test after slicing:', list_test)

    # 切片清空
    list_test[:] = []
    print('list_test[:] = [] -> :', list_test)

    list_test2 = [0, 1, 2, 3, 4]
    print('list_test2[:2] = ', list_test2[:2])
    print('list_test2[:-3] = ', list_test2[:-3])



