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
        # * 解包元组
        >>> list(range(3, 6))            # normal call with separate arguments
        [3, 4, 5]
        >>> args = [3, 6]
        >>> list(range(*args))            # call with arguments unpacked from a list
        [3, 4, 5]

        # ** 解包字典
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

def ask_ok(ok : str = 'yes', retries = 4, reminder = 'Please try again!'):
    while True:
        # ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)

def list_append(ele : int = 10, act_list : list = []): 
    '''
    ele: append value
    list: tobe append list
    '''
    act_list.append(ele)
    return act_list

def list_append_fix(ele, list = None):
    if list == None:
        list = []
    list.append(ele)
    return list

def variable_param_test(*input_tuple):
    return input_tuple

def variable_param_test2(**input_dict):
    return input_dict

if __name__ == '__main__':
    '''
    1、if else 条件语句
    '''
    # val = int(input("please input the val:"))
    val = 12
    if val == 0:
        print("input val equal zero, val = ", val)
    elif val > 0:
        print("input val > zero, vlvala = ", val)
    elif val < 0:
        print("input val < zero, val = ", val)

    print('------------------------')

    '''
    2、for 循环语句
    '''
    users = {'Hans': 'active', 'Éléonore': 'inactive', '景太郎': 'active', 'harry': 18}
    user_ages = {'Hans': 12, 'Éléonore': 18, '景太郎': 24}

    # 遍历字典
    for user, status in users.items():
        print('user:', user, 'status:', status)
    print('***')

    for user, status in users.items():
        print(f'user: {user}, status: {status}')
    print('***')

    for user, age in user_ages.items():
        print('user: %s, age: %d' % (user, age))
    print('***')

    # 删除 users 字典的指定元素
    for user, status in users.copy().items():
        if status == 'inactive':
            print("-- del the key:", user)
            del users[user]

    # 新建字典保存删除指定元素的字典
    active_user = {}
    for user, status in users.items():
        if status != 'inactive':
            active_user[user] = status

    print('------------------------')

    '''
    3、break, continue, for - else 子句: for 循环语句结束一直都没有break的情况会走入 else 子句
    '''
    print("range(2, 10):", list(range(2, 10)))
    print("range(2, 2):", list(range(2, 2)))

    for i in range(2, 10):
        for j in range(2, i):
            print('i:', i, 'j:', j)
            if i % j == 0:
                print(i, 'equals', j, '*', i//j)
                break
        else:
            print(i, 'is a prime number')

    '''
    4、range 用于生成等差数列 - range(start, end, delta), 数列包含start, 不包含end
       range 返回一个可迭代对象，可以用于sum函数参数
    '''
    print("list(range(5)):", list(range(5)))
    print("tuple(range(5)):", tuple(range(5)))

    my_list = [1, 2, 3, 4, 'apple', 5.6]

    for i in range(len(my_list)) :
        print('my_tuple[%d] = ' % i, my_list[i])

    '''
    5、pass 语句：不执行任务动作
    '''
    # 常用于创建最小类
    class MyEmptyClass:
        pass    

    # 忙等待
    # while True:
    #     pass  # Busy-wait for keyboard interrupt (Ctrl+C)
    
    '''
    6、match 语句：
    '''

    print('---------------------')

    '''
    7、函数 def, 函数没有 return 时也有返回值, 返回值为 None
    7.1、默认值参数 - 函数行参后加 = 默认参数
        def ask_ok(prompt, retries=4, reminder='Please try again!'):
    '''
    print('7.1 *********************')

    print("ask_ok('input yes or no: ') = ", ask_ok(reminder = "[LC] Please try again!"))

    # !!! 默认值为列表、字典或类实例等可变对象时，默认值只计算一次，默认值多次调用会存在共享内容
    ele0, ele1 = 0, 'ele1'
    ele2 = [12, 18, 'ele2']

    print("list_append(ele0):", list_append(ele0))
    print("list_append(ele1):", list_append(ele1))
    print("list_append(ele2):", list_append(ele2))

    print("list_append_fix(ele0):", list_append_fix(ele0))
    print("list_append_fix(ele1):", list_append_fix(ele1))
    print("list_append_fix(ele2):", list_append_fix(ele2))

    '''
    7.2、关键字参数
        调用的时候关键字参数和位置参数，每一个行参选择一个使用即可

        函数行参中: / 前的参数只能使用位置参数, * 后的参数只能使用关键字参数，其他参数使用位置参数还是关键字参数都可以
            def f(pos1, pos2, /, pos_or_kwd, *, kwd1, kwd2):
                    -----------    ----------     ----------
                        |             |                  |
                        |        Positional or keyword   |
                        |                                - Keyword only
                        -- Positional only
    '''
    print('7.2 *********************')

    print("list_append(act_list = [1, 2, 3]):", list_append(act_list = [1, 2, 3]))

    '''
    7.3、可变参数
        - 最后一个参数为 **name 时接收一个字典, 
        - 最后一个参数为 *name 形参接收一个元组 (*name(元组) 必须在 **name(字典) 前面)
        - 可变参数后面的参数只能使用关键字参数传递
    '''
    print('7.3 *********************')

    print('variable_param_test(ele0, ele1, ele2):', variable_param_test(0, 'test', 25, 'test_29'))
    print('variable_param_test2(ele0, ele1, ele2):', variable_param_test2(Hans=12, Éléonore='five', 景太郎=24))

    '''
    7.4、函数注解和文档字符串
        函数注解:
        def list_append(ele:int = 10, act_list:list = []) -> str: 
    '''
    print('7.4 *********************')

    print('list_append.__doc__:', list_append.__doc__)
    print('list_append.__annotations__:', list_append.__annotations__)

    '''
    7.5、解包实参列表
        调用函数时，* 操作符用于解包 元组 | 列表 用于函数的实参
                  ** 操作符用于解包 字典 用于函数的实参
    '''
    print('7.5 *********************')

    limit = [2, 8]    
    print("range(*limit):", list(range(*limit)))

    param_dict = {'ele': 20, 'act_list' : [1, 3, 5, 7] }
    param_dict2 = {'act_list' : [1, 3, 5, 7] }

    print("list_append(**param_dict): ", list_append(**param_dict))
    print("list_append(**param_dict2): ", list_append(**param_dict2))

    '''
    7.6、lambda 表达式, 用法: lambda a, b: a+b
    '''
    print('7.6 *********************')

    add = lambda a, b : a + b
    print("add(1, 3):", add(1, 3))

    hell_world = lambda: "Hello, World!"
    print("hell_world():", hell_world())

    # !!! map()、filter()、sorted() 配合 lambda 使用
    numbers = [(4, 2), (3, 8), (2, 10), (1, 6), (6, 5), (5, 1)]
    sorted_numbers = sorted(numbers, key=lambda x: x[1], reverse=True) # 按照元组下标为1的元素，reverse: 降序排列，

    print("sorted_numbers:", sorted_numbers)
    print('list(map(lambda x: x ** 2, sorted_numbers)):', list(map(lambda x: (x[0] ** 2, x[1] ** 2), sorted_numbers)))
    print('list(filter(lambda x: x % 2 == 0, sorted_numbers)):', list(filter(lambda x: x[1] % 2 == 0, sorted_numbers)))
    
    print('---------------------')

    fs = frozenset([1, 2, 3])
    values = ["first", "second", "third"]

    print('fs:', fs)
    print('values:', values)

    # 使用 zip() 和字典推导式
    my_dict = {key: value for key, value in zip(fs, values)}
    print(my_dict)

