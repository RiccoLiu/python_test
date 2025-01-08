#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Person:
    def __init__(self, name, age):
        self._name = name
        self._age = age

    def to_dict(self):
        return {'name': self._name, 'age': self._age}

# 自定义异常
class MyCustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

if __name__ == '__main__':

    '''
        []: 列表 list 
            empty_list = []
            no_empty_list = [value1, value2]

        (): 元组 tuple
            empty_tuple = ()
            no_empty_tuple = (value1, value2)
            
            one_tuple = (value1,)
            one_tuple = value1, 

        {key : val}: 字典 dict    
            empty_dict = {}
            no_empty_dict = {key1:value1, key2:value2}

        {}: 集合 set, 可以用序列初始化 set(序列) 
            empty_set = set()
            no_empty_set = {value1, value2}

        frozenset: 只读集合，需要用序列初始化 frozenset(序列)
            empty_frozenset = frozenset()
            no_empty_frozenset = frozenset(value1, value2)
    '''

    a = 'hello', # 创建元组  ('hello') !! 有逗号就是元组，无逗号就是字符串
    print(type(a))

    a = ('hello', ) # 创建元组  ('hello') !! 有逗号就是元组，无逗号就是字符串
    print(type(a))

    a = 'hello'   # 创建的字符串 hello 
    print(type(a))

    a = ('hello') # 创建的字符串 hello 
    print(type(a))

    a = {'hello'} # 创建的集合 {'hello'}
    print(type(a))

    a = {'hello',} # 创建的集合 {'hello'}
    print(type(a))

    a = set('hello') # 创建的集合 {'e', 'o', 'h', 'l'} !! 这里可以把 'hello' 认为是一个序列，对这个序列取集合后就是一个个的不重复的子母
    print(type(a))

    a = frozenset('hello') # 创建的不可变集合 {'e', 'o', 'h', 'l'} 
    print(type(a))

    '''
        字符串格式化
    '''
    s = "Hello, world!"
    print(str(s))  # 输出: Hello, world!
    print(repr(s))  # 输出: 'Hello, world!'

    print('--------------')

    '''
        读写文件：
    '''
    dict_test = {1:'first', 2:'second'}
    tuple_test = (1, 'first', 2, 'second')
    list_test = [1, 'first', 2, 'second']

    lines = ["First line\n", "Second line\n", "Third line\n"]

    with open('test.txt', 'w+') as f:
        f.write('--- lane start ---\n')

        f.write(str(dict_test) + '\n')
        f.write(str(tuple_test) + '\n')
        f.write(str(list_test) + '\n')

        f.writelines(lines)
        f.write('--- lane end ---\n')

    try:
        with open('test.txt', 'r') as f:
            content = f.read()
            print(content, end='')

            f.seek(0)

            line_idx = 0
            for line in f.readlines():
                print(f'line_idx:{line_idx}, {line}', end='')
                line_idx += 1

    except FileNotFoundError:
        print("文件未找到")
    except PermissionError:
        print("没有权限访问文件")

    print('---------------------------')

    '''
        使用json保存结构化数据
            json.dumps()
                indent: 
                    控制缩进级别
                    
                separators: 
                    控制分割符, 默认使用', '和': '
      
                sort_keys: 
                    对字典进行排序

                ensure_ascii:
                    默认ensure_ascii 为True即都是ascii字符,如果不是ascii字符(中文)会被转义(乱码)
                    
                default:
                    处理不可序列化对象
    '''

    import json

    data = {"name": "Alice", "age": 30, "city": "北京"}

    json_string = json.dumps(data)
    print(json_string)

    json_string = json.dumps(data, separators=(",", ":"))
    print(json_string)

    json_string = json.dumps(data, indent = 4, separators=(',', ' --> '), sort_keys=True, ensure_ascii=False)
    print(json_string)

    with open('test.json', 'w+') as f:
        print(f'save_json: {data}')
        json.dump(data, f, ensure_ascii=False) # 有中文

    load_json = {}
    with open('test.json', 'r') as f:
        load_json = json.load(f)

    print(f'load_json: {load_json}')        

    mike = Person('mike', 15)
    json_mike = json.dumps(mike, default=lambda m : m.to_dict())

    print(f'json_mike: {json_mike}')

    '''
        字符串打印对齐：
            str().rjust(10): 给定宽度下，进行右对齐
            str().ljust(10): 给定宽度下，进行左对齐
            str().center(10): 给定宽度下，进行中间对齐
    '''
    for x in range(1, 11):
        print(str(x).rjust(2), str(x*x).rjust(3), end=' ')
        print(str(x*x*x).rjust(4))

    for x in range(1, 11):
        print(str(x).ljust(2), str(x*x).ljust(3), end=' ')
        print(str(x*x*x).ljust(4))

    print('--------------------------')

    '''
        异常
            try:
                执行可能抛出异常的代码
            except:
                ValueError:
                    无效值异常
                ZeroDivisionError:
                    除以 0 异常
                Exception:
                    可以匹配任何异常
            else:
                没有任何异常可以执行这里的代码
            finally:
                无论有没有异常都执行这里的代码

            raise:
                抛出异常
 
            Exception:
                # 自定义异常
                class MyCustomError(Exception):
                    def __init__(self, message):
                        self.message = message
                        super().__init__(self.message)     
    '''

    result = None

    try:
        x = int(input("请输入一个数字: "))
        result =  10 / x
    # except ValueError as ve:
    #     print(f'无效的值: {ve}')
    except ZeroDivisionError as ze:
        print(f'除以零错误: {ze}')
    except Exception as e:
        print(f'Exception 可以匹配任何异常')
    else:
        print('没有异常可以执行到这里..')
    finally:
        print('无论是否有异常都可以执行到这里..')

    try:
        raise MyCustomError("这是一个自定义的错误")
    except MyCustomError as e:
        print(f"捕获到自定义异常: {e}")
