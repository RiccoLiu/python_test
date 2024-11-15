#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
一、控制流语句：
1、if else 条件语句
    x = int(input("Please enter an integer: "))
    if x < 0:
        x = 0
        print('Negative changed to zero')
    elif x == 0:
        print('Zero')

2、for 循环语句
    迭代多项集合的时候修改集合是比较困难的，更方便的是迭代多项集的拷贝时创建新的多项集：

    users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active'}

    # 字典循环查找删除这里要用copy, 否则迭代会有问题
    for user, status in users.copy().items():
        if status == 'inactive':
            del users[user]

    # 字典迭代过程创建新的删除指定 Key 的字典
    active_users = {}
    for user, status in users.items():
        if status == 'active':
            active_users[user] = status
        
3、range 用于生成等差数列 - range(start, end, delta), 数列包含start, 不包含end
    3.1、生成等差数列 0, 1, 2, 3, 4
        for i in range(5):
            print(i)

    3、2、等差数列转 列表
        a = list(range(10, 100, 25));     

    3.2、range 配合len 通过下标访问 列表
        a = ['Mary', 'had', 'a', 'little', 'lamb']
        for i in range(len(a)):
            print(i, a[i])

    3.3、range返回的是一个可迭代对象, 可以用做sum函数参数

4、break, continue, 循环语句的 else 子句 
    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(n, 'equals', x, '*', n//x)
                break
        else:
            # loop fell through without finding a factor
            print(n, 'is a prime number')

    # else 子句属于 for, 如果没有发生 break 会执行 else 子句

5、pass; 语句不执行任何动作
    while True:
        pass  # Busy-wait for keyboard interrupt (Ctrl+C)
    
    // 常用于创建最小类
    class MyEmptyClass:
        pass
        
    // 用作函数或条件语句体的占位符, 类似 TODO
    def initlog(*args):
        pass   # 记得之后实现这个操作

6、match; 接受一个表达式，把他的值与其他一个或多个值对比，类似 switch cast
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _: # _ 通配符必定会匹配成功
            return "Something's wrong with the internet"

    # 
    case 401 | 403 | 404:
        return "Not allowed"

7、定义函数
    函数没有 return 时也有返回值, 返回值为 None

    7.1、默认值参数
        def ask_ok(prompt, retries=4, reminder='Please try again!'):
            while True:
                ok = input(prompt)
                if ok in ('y', 'ye', 'yes'):
                    return True
                if ok in ('n', 'no', 'nop', 'nope'):
                    return False
                retries = retries - 1
                if retries < 0:
                    raise ValueError('invalid user response')
                print(reminder)
        
        !!! 默认值为列表、字典或类实例等可变对象时，默认值只计算一次，默认值多次调用会存在共享内容
        def f(a, L=[]):
            L.append(a)
            return L

        print(f(1)) -> [1]
        print(f(2)) -> [1, 2]
        print(f(3)) -> [1, 2, 3]

        # 如果不希望默认值共享，可以按照如下编码：
        def f(a, L=None):
            if L is None:
                L = []
            L.append(a)
            return L
    
    7.2、关键字参数
        def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
            print("-- This parrot wouldn't", action, end=' ')
            print("if you put", voltage, "volts through it.")
            print("-- Lovely plumage, the", type)
            print("-- It's", state, "!")

        合法调用：
            parrot(1000)                                          # 1 位置参数
            parrot(voltage=1000)                                  # 1 关键字参数
            parrot(voltage=1000000, action='VOOOOOM')             # 2 关键字参数
            parrot(action='VOOOOOM', voltage=1000000)             # 2 关键字参数
            parrot('a million', 'bereft of life', 'jump')         # 3 位置参数
            parrot('a thousand', state='pushing up the daisies')  # 1 位置参数, 1 关键字参数
        
        非法调用：
            parrot()                     # 缺少必选参数
            parrot(voltage=5.0, 'dead')  # 关键字参数后是非关键字参数
            parrot(110, voltage=220)     # 关键字参数与位置参数冲突
            parrot(actor='John Cleese')  # 没有这个关键字    
            
        # 最后一个参数为 **name 时接收一个字典, *name 形参接收一个 元组 (*name 必须在 **name 前面)
        def cheeseshop(kind, *arguments, **keywords):
            print("-- Do you have any", kind, "?")
            print("-- I'm sorry, we're all out of", kind)
            for arg in arguments:
                print(arg)
            print("-" * 40)
            for kw in keywords:
                print(kw, ":", keywords[kw])
        # 调用
        cheeseshop("Limburger", "It's very runny, sir.",
            "It's really very, VERY runny, sir.",
            shopkeeper="Michael Palin",
            client="John Cleese",
            sketch="Cheese Shop Sketch")
    
    7.3、特殊参数
        仅限位置形参应放在 / （正斜杠）前; 仅限关键字参数形式传递该形参，应在参数列表中第一个 仅限关键字 形参前添加 *。

        def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
              -----------    ----------     ----------
                  |             |                  |
                  |        Positional or keyword   |
                  |                                - Keyword only
                  -- Positional only
     
    7.4、函数的可变参数
        末尾的 *args 可变参数作为元组使用；        
        def write_multiple_items(file, separator, *args):
            file.write(separator.join(args))
        
        !!! 可变参数后的参数只能使用关键字参数传递

    7.5、解包实参列表
        # * 接包元组
        >>> list(range(3, 6))            # normal call with separate arguments
        [3, 4, 5]
        >>> args = [3, 6]
        >>> list(range(*args))            # call with arguments unpacked from a list
        [3, 4, 5]

        # ** 接包字典
        >>> def parrot(voltage, state='a stiff', action='voom'):
        ...     print("-- This parrot wouldn't", action, end=' ')
        ...     print("if you put", voltage, "volts through it.", end=' ')
        ...     print("E's", state, "!")
        ...
        >>> d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
        >>> parrot(**d)
        -- This parrot wouldn't VOOM if you put four million volts through it. E's bleedin' demised !

    7.6、lamda表达式
        使用方式：lambda a, b: a+b 函数返回两个参数的和

        def make_incrementor(n):
            return lambda x: x + n

    7.7、函数注解
        标注 以字典的形式存放在函数的 __annotations__ 属性中, 可以打印出来看下

        def f(ham: str, eggs: str = 'eggs') -> str:
            print("Annotations:", f.__annotations__)
            print("Arguments:", ham, eggs)
            return ham + ' and ' + eggs

        >>>> f('spam')
        Annotations: {'ham': <class 'str'>, 'return': <class 'str'>, 'eggs': <class 'str'>}
        Arguments: spam eggs
        'spam and eggs'

    7.8、文档字符串 docstring
        def my_function():
            """Do nothing, but document it.

            No, really, it doesn't do anything.
            """
            pass

        # 打印函数的 docstring
        print(my_function.__doc__)
'''

if __name__ == '__main__':

    while True:
        pass  # Busy-wait for keyboard interrupt (Ctrl+C)