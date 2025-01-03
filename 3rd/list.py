#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ == '__main__':
    list_num = [0, 1, 2, 3, 4, 5]
    print('len(list_num) = ', len(list_num), 'list_num:', list_num)

    list_str = ['a', 'b', 'c', 'd', 'e']
    print('len(list_str) = ', len(list_str), 'list_str:', list_str)

    print('*****')
    '''
        list.copy(): 返回列表的浅拷贝, 列表中的不可变元素会拷贝到新列表，可变元素会把指针返回新的列表中, 即改变新列表中的可变元素也会影响到旧列表
                list_copy = list.copy() 等价于  list_copy = list[:]
    '''
    list_num_str = list_num.copy()
    list_num_str.append(list_str)

    print('len(list_num_str) = ', len(list_num_str), 'list_num_str:', list_num_str)

    list_num_str_copy = list_num_str.copy()
    list_num_str_copy[0] = 10       # 修改 list_num_str_copy 第一个元素是不可变元素，不会影响原来的列表
    list_num_str_copy[-1][0] = 100  # 修改 list_num_str_copy 最后一个元素是可变元素，相当于修改指针会影响到原来的列表

    print('len(list_num_str_copy) = ', len(list_num_str_copy), 'list_num_str_copy:', list_num_str)
    print('len(list_num_str) = ', len(list_num_str), 'list_num_str:', list_num_str)

    print('*****')

    '''
        list.append(x) :
            在列表末尾添加一个元素，相当于 list[len(list):] = [x] 。
    '''
    list_num1 = list_num.copy()
    list_num2 = list_num[:]

    list_num1.append([6, 7, 8])
    print('len(list_num1) = ', len(list_num1), 'list_num1:', list_num1)

    list_num2[len(list_num2):] = [[6, 7, 8]] 
    print('len(list_num2) = ', len(list_num2), 'list_num2:', list_num2)

    print('*****')

    '''
        list.extend(iterable): 
            将另一个可迭代对象（如列表、元组或字符串）中的元素逐一添加到原列表的末尾, 
            相当于  list[len(alist):] = iterable
    '''
    list_num3 = list_num.copy()
    list_num4 = list_num[:]

    list_num3.extend([6, 7, 8])
    print('len(list_num3) = ', len(list_num3), 'list_num3:', list_num3)

    list_num4[len(list_num4):] = [6, 7, 8]
    print('len(list_num4) = ', len(list_num4), 'list_num4:', list_num4)

    print('*****')

    '''
        list.insert(idx, x) : 在指定下标前插入元素。
            a.insert(0, x): 列表头插入元素
            a.insert(len(a), x): 列表尾插入元素 等同于 a.append(x)
    '''
    list_num5 = list_num[:]
    list_num5.insert(0, -1)
    list_num5.insert(len(list_num5), list_num5[-1] + 1)
    print('len(list_num5) = ', len(list_num5), 'list_num5:', list_num5)

    list_num5_copy = list_num5[:]

    list_num5.insert(len(list_num5), list(range(list_num5[-1] + 1, list_num5[-1] + 6)))
    print('len(list_num5) = ', len(list_num5), 'list_num5:', list_num5)

    # 如果不希望新插入的列表作为单个新元素插入时，可以使用切片赋值 或者 循环插入
    list_num5_copy[len(list_num5_copy) :] = list(range(list_num5_copy[-1] + 1, list_num5_copy[-1] + 6))
    print('len(list_num5_copy) = ', len(list_num5_copy), 'list_num5_copy:', list_num5_copy)

    '''
        list.remove(x): 根据值删除元素(删除列表第一个值为x的元素)，未找到指定元素时，触发 ValueError 异常。
    '''
    list_num6 = list_num[:]
    list_num6.append(list_str)
    list_num6.append(list_str)
    print('len(list_num6) = ', len(list_num6), 'list_num6:', list_num6)

    list_num6.remove(3)
    print('len(list_num6) = ', len(list_num6), 'list_num6:', list_num6)

    list_num6.remove(list_str)
    print('len(list_num6) = ', len(list_num6), 'list_num6:', list_num6)

    '''
        list.pop(idx): 弹出指定下标的元素，如果不写idx,默认弹出末尾元素
    '''
    list_num6.pop(0)
    print('len(list_num6) = ', len(list_num6), 'list_num6:', list_num6)

    back = list_num6.pop(-1)
    print('len(list_num6) =', len(list_num6), 'list_num6:', list_num6, 'back:', back)

    '''
        list.clear(): 清空列表操作
            等价于 del a[:] 或者 a[:] = []
    '''
    list_num6.clear()
    print('len(list_num6) =', len(list_num6), 'list_num6:', list_num6)

    '''
        list.sort(*, key=None, reverse=False): 原地排序，默认升序;
            sorted: 排序可以用于其他可变元素排序，不会跟原来的序列，返回排序过的新序列

            列表中只有数字可以排序，只有字符串可以排序，混合数字和字符串不可以排序
    '''
    list_num6 = list_num.copy()
    list_num6.extend([6.8, 7.4, 8, 10])

    list_num6.sort(reverse=True)
    print('len(list_num6) =', len(list_num6), 'list_num6:', list_num6)

    list_num6 = list_str[1:]
    list_num6.sort(reverse=True)
    print('len(list_num6) =', len(list_num6), 'list_num6:', list_num6)

    '''
        list.reverse(): 列表翻转
    '''
    list_num6.reverse()
    print('len(list_num6) =', len(list_num6), 'list_num6:', list_num6)

    '''
        list.count(x): 统计列表出现x的次数
    '''
    list_num7 = list_num + [list_str] + [list_str] + [list_str] + list_num

    print('len(list_num7) =', len(list_num7), 'list_num7:', list_num7)
    print('list_num7.count(list_str):', list_num7.count(list_str), 'list_num7.count(list_num7[0]):', list_num7.count(list_num7[0]))

    '''
        list.index(x): 返回列表中第一个元素为x的下标, 可以指定查找的起始和终止下标, list(x, start_idx, end_idx), 如果没有找到会触发 ValueError 异常   
    '''
    list_str_idx = list_num7.index(list_str)
    print('list_str_idx: ', list_str_idx)

    list_str_idx = list_num7.index(list_str, 7, len(list_num7))
    print('list_str_idx: ', list_str_idx)

    '''
        使用 - 连接列表的元素
    '''
    date = ['2024-07-08', '2024-08-09', 'test']
    date_str = '-'.join(date)
    print(f'date = {date}, data_str: {date_str}')
    