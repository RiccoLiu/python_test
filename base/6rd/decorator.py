#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# def decorator(func):
#     '''
#         装饰器这样定义时，会在装饰器应用是立即执行下面的函数，这不是我们想要的效果
#     '''

#     print("---- decorator start ----")
#     func()
#     print("---- decorator end ----")

# @decorator
# def my_function():
#     print('--- call my_function ----')
    
def decorator2(func):
    def wrapper():
        print("---- decorator2 start ----")
        func()
        print("---- decorator2 end ----")
    return wrapper

@decorator2
def my_function2():
    print('--- call my_function2 ----')

def singleton(cls):
    '''
        单例模式
    '''

    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class Circle:
    count = 0 # 类属性

    def __init__(self, radius):
        self.__radius = radius
        Circle.count += 1
    
    # 类方法: 第一个参数是cls, 方法内可以访问类属性和类方法，不能访问实例属性和方法，可以通过类名或者实例调用
    @classmethod
    def call_cls_method(cls):
        print(f'---- call classmethod, count = {cls.count}----')

    # 类静态方法: 没有参数，方法内不可以访问类属性和类方法，也不能访问实例属性和方法，可以通过类名或者实例调用
    @staticmethod
    def call_static_method():
        print('---- call staticmethod ----')

    # 属性：将实例方法转换为实例属性
    @property
    def radius(self):
        return self.__radius

    # 控制实例属性的 setter 的行为
    @radius.setter
    def radius(self, value):
        if value < 0:
            # raise ValueError("Radius must be positive")
            print('set radius value failed, value:', value)
        else:
            print('set radius value:', value)
            self.__radius = value

    # 控制实例属性的 deleter 的行为
    @radius.deleter
    def radius(self):
        print("Deleting radius")
        del self._radius
    
    # 动态计算实例的属性
    @property
    def area(self):
        return 3.14 * self.__radius * self.__radius

    # 实例方法
    def func(self):
        print("----- call func ----\n")

    # 类的工厂方法
    @classmethod
    def create_circle(cls, radius):
        return cls(radius)
    

if __name__ == '__main__':

    '''
        @ 用法：
            函数装饰器:
                def decorator2(func):
                    def wrapper():
                        func()
                    return wrapper
                
                @decorator2
                def my_function2():
                    print('--- call my_function2 ----')

            类装饰器：
                def singleton(cls):
                    instances = {}

                    def get_instance(*args, **kwargs):
                        if cls not in instances:
                            instances[cls] = cls(*args, **kwargs)
                        return instances[cls]

                    return get_instance
                   
            类方法的装饰器
                类方法:
                    @classmethod
                        # 类方法: 第一个参数是cls, 方法内可以访问类属性和类方法，不能访问实例属性和方法，可以通过类名或者实例调用
                        @classmethod
                        def call_cls_method(cls):
                            print('---- call classmethod ----')

                        通常用于类的工厂方法, 计数器等
                        
                类静态方法:
                    @staticmethod
                        # 类静态方法: 没有参数，方法内不可以访问类属性和类方法，也不能访问实例属性和方法，可以通过类名或者实例调用
                        @staticmethod
                        def call_static_method():
                            print('---- call staticmethod ----')
    
                        通常用于处理类无关但放在类中的辅助的逻辑
                        
                    类方法与类静态方法对比：
                        特性	        @classmethod	                                    @staticmethod
                        第一个参数	   cls(类本身)	                                      无参数（不传递类或实例）
                        访问权限	   可以访问类属性和类方法，不能访问实例属性和实例方法	     不能访问类属性、类方法、实例属性或实例方法
                        用途	      用于处理与类本身相关的逻辑（如工厂方法、计数器等）	    用于处理与类无关但放在类中的辅助逻辑
                        调用方式	   可以通过类或实例调用	                                可以通过类或实例调用
                        
                类属性：
                    @property
                        # 属性: 将实例方法转换为实例属性
                        @property
                        def area(self):
                            return 3.14 * self.__radius * self.__radius
                        
                        # @property 配合 setter, deleter 的使用控制实例属性的设置和删除行为    
                     
            矩阵运算

    '''
    f = my_function2

    print('----------------- after f ----')
    f()

    print('------------------ class decorator test start -----')

    little_circle = Circle.create_circle(2.0)

    little_circle.call_cls_method()
    Circle.call_cls_method()

    little_circle.call_static_method()
    Circle.call_static_method()

    print(f'little_circle.radius:{little_circle.radius}')
    print(f'little_circle.area:{little_circle.area}')

    print(little_circle.count)
    print(Circle.count)

    little_circle.radius = -1.0
    little_circle.radius = 4.0

    print(f'little_circle.radius:{little_circle.radius}')
    print(f'little_circle.area:{little_circle.area}')

    print('------------------ class decorator test end -----')

