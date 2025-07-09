#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import weakref

class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def print_info(self):
        print(f'name:{self.name}, age:{self.age}')

def on_finalize():
    print("hurry has been garbage collected!")

class CacheData:
    def __init__(self, name):
        self.name = name

class Cache:
    def __init__(self):
        self._cache = {}

    def Put(self, key, data):
        self._cache[key] = data

    def Get(self, key):
        if key in self._cache:
            return self._cache[key]
        return None

    def GetWeak(self, key):
        if key in self._cache:
            return self._cache[key]()

    def PutWeak(self, key, data):
        self._cache[key] = weakref.ref(data)

if __name__ == '__main__':

    '''
        weakref:
            弱引用只能用于自建类型，对于内置的容器(list, dict, tuple)等,不可以直接使用ref 弱引用。

            ref:
                弱引用对象, 如果原始对象已经被回收, 返回None, 弱引用对象不可以直接访问引用对象内的属性。
            proxy:
                代理对象, 如果原始对象已经被回收, 返回None, 代理对象可以直接访问对象内的属性。
            
            finalize:
                注册回调函数，在引用对象被回收时调用
            
            应用场景：
                用于缓存场景，特别是当你不想强引用缓存对象，从而避免内存泄漏或保持不必要的数据时           
    '''

    mike = Student('mike', 18)
    mike.print_info()

    weakref_mike = weakref.ref(mike)
    print(weakref_mike())

    del mike
    print(weakref_mike())

    hurry = Student('hurry', 20)
    weakref.finalize(hurry, on_finalize)

    weakref_hurry = weakref.ref(hurry)

    # 报错
    # weakref_hurry.print_info()

    weakref_hurry().print_info()

    weak_proxy_hurry = weakref.proxy(hurry)
    weak_proxy_hurry.print_info()

    weakref_hurry().age = 30
    hurry.print_info()

    weak_proxy_hurry.age = 40
    hurry.print_info()

    del hurry

    # print(weak_proxy_hurry)
    print(weakref_hurry())

    print('--------- weak cache test ------------')

    cache_data = CacheData("test_data")
    print(cache_data)

    cache = Cache()

    cache.Put('test_data', cache_data)
    print(cache.Get('test_data'))

    # 这里只是cache_data的引用记数-1，由于在cache的字典里是强引用，cache_data引用计数并未清0，内存也不会释放
    del cache_data

    # cache_data的内存没有释放，这里还可以get出来
    print(cache.Get('test_data'))

    print('--------- use cache weak ----')

    cache_data2 = CacheData("test_data2")
    print(cache_data2)

    cache2 = Cache()

    cache2.PutWeak('test_data2', cache_data2)
    print(cache2.GetWeak('test_data2'))

    # cache里存的都是弱引用，这里引用记数-1后，引用计数清0，内存已经释放
    del cache_data2

    # 内存释放后，这里 get 出来 None
    print(cache2.GetWeak('test_data2'))



