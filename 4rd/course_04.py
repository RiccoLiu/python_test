#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

@dataclass
class DataFormat:
    id: int
    name: str


'''
    结构体：
        from dataclasses import dataclass

        @dataclass
        class DataFormat:
            id: int
            name: str
'''

'''
    nonlocal:
        嵌套函数更改父函数的变量时使用
    global:
        函数内改全局变量时使用
'''
x = 1

def outer():
    out_x = 10

    def middle():
        middle_x = 100

        nonlocal out_x
        out_x = 100 # 更改嵌套函数的父函数变量成功

        def inner():
            inner_x = 1000
            
            global x

            x = 1000 # 更改全局变量成功
            middle_x = 1000 # 更改嵌套函数的父函数变量失败，这里的 middle_x 是一个新的变量
        
        inner()

        print('after call inner, x = ', x, 'middle_x = ', middle_x)
    middle()
    print('after call middle, x = ', x, 'out_x =', out_x)

'''
    类：
        1、构造函数 __init__, 每个类只有一个构造函数
        2、类的方法必须有 self 参数
        3、类属性 & 实例属性
            类属性相当于C++的静态成员，访问时可以用 类名.类属性(Person.name) 访问，也可以用 实例对象.__class__.类属性(jack.__class__.name) 访问
            类属性和实例属性同名时, 通过 self 赋值的都是实例属性
            允许 类外部 给实例添加属性, 类属性或者实例属性都可以。
            
        4、类的方法可以访问全局作用域, 但类本身不是全局作用域(外部不可以直接访问类的实例属性)
        5、继承 - 继承使得子类能够复用父类的属性和方法，避免重复代码。
            class DerivedClassName(modname.BaseClassName):

            5.1、子类可以用supper()调用父类的方法，也可以指定 BaseClass.Method 调用父类的方法      
            5.2、方法重写, 类的所有方法均是 virtaul 即都可以重写
            5.3、多重继承 MRO

        6、特殊变量
            保护变量: 变量前一个下划线，约定开发者不应该外部访问这个变量，但实际也是可以访问
            私有变量: 变量前两个下划线，触发 名称重整 外部不可以直接访问，应该通过公有方法访问(getter, setter)
                jack_boy.__private_var ---> jack_boy._Boy__private_var
            
        7、结构体
            from dataclasses import dataclass

            @dataclass
            class DataFormat:
                id: int
                name: str
                
        函数：
            isinstance(obj, class): 检查对象是否是类的实例
            issubclass(DerivedClass, BaseClass) : 检查继承关系 
'''

class Person:
    """ A Person Class Test """
    name = 'Person' # 类属性, 类似 C++ 类的静态变量

    def __init__(self, name = 'default', age = 0, grade = 0):
        self.name = name # 实例属性
        self.age = age
        self.grade = grade
        self.toy = []

    def Play(self) :
        print(f'{self.name} play')

    def AddToy(self, toy):
        self.toy.append(toy)

    def AddTwoToy(self, first_toy, second_toy):
        self.toy.append(first_toy)
        self.toy.append(second_toy)

    def Greet():
        print(f'global x = {x}')

class Boy(Person):
    ''' Inherited from Person '''

    def __init__(self, name = 'default', age = 0, grade = 0, sport = 'runing'):
        Person.__init__(self, name, age, grade)
        
        self.sport = sport
        self._protected_var = 'protected_var'
        self.__private_var = 'private_var'

    def Play(self):
        super().Play()
        print(self.sport)


class Reverse:
    """Iterator for looping over a sequence backwards."""
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]


'''
    迭代器 -  定义 __iter__(self), __next__(self), __next__(self) 将引发 StopIteration 异常来通知终止 for 循环

        class Reverse:
            """Iterator for looping over a sequence backwards."""
            def __init__(self, data):
                self.data = data
                self.index = len(data)

            def __iter__(self):
                return self

            def __next__(self):
                if self.index == 0:
                    raise StopIteration
                self.index = self.index - 1
                return self.data[self.index]

        生成的迭代器不可以直接使用，可以使用 for 循环取出每个迭代器的值，也可以用 list 转换成列表，比如 range(5) 也是生成的迭代器
                
    生成器 - yield 使函数变成一个迭代器，迭代器可以逐个返回多个值，而不需要一次把所有值加载到内存中。

        生成器表达式：
'''

# yield 生成反向迭代器
def Reverse(data):
    for idx in range(len(data) - 1, -1, -1):
        yield data[idx]

if __name__ == '__main__':
    '''
        命令空间:
            global:
            nonlocal:
    '''
    outer()
    print('after call outer, x = ', x)

    # 类的实例化
    jack = Person('jack', 8, 'grade 1')
    print(f'name: {jack.name}, age: {jack.age}, grade: {jack.grade}, class.name:{Person.name}')
    
    # 修改实例属性
    jack.age = 10
    print(f'name: {jack.name}, age: {jack.age}, grade: {jack.grade}')

    # 修改类属性
    print('Person.name:', Person.name)
    print('jack.__class__.name:', jack.__class__.name)

    jack.__class__.name = 'jack person'

    print('Person.name:', Person.name)
    print('jack.__class__.name:', jack.__class__.name)

    # 类的方法绑定
    jack_play = jack.Play
    for i in range(2) :
        jack_play()

    # 类外添加属性, 原本类的实例属性没有 address, 类属性没有 obj
    jack.address = 'shen yang'
    jack.__class__.obj = 'obj0'
    
    print(f'name: {jack.name}, age: {jack.age}, grade: {jack.grade}, address:{jack.address}, obj:{jack.__class__.obj}, class.name:{Person.name}, class.obj:{Person.obj}')

    # 继承
    jack_boy = Boy(jack.name, jack.age, jack.grade, 'basketball')
    print(f'name: {jack_boy.name}, age: {jack_boy.age}, grade: {jack_boy.grade}, obj:{jack.__class__.obj}, class.name:{Person.name}, class.obj:{Person.obj}')

    jack_boy.Play()

    # issubclass(DerivedClass, BaseClass) : 检查类的继承关系
    print(f'issubclass(Boy, Person) = {issubclass(Boy, Person)}')

    # isinstance(obj, class) : 检查对象是否是类实例
    print(f'isinstance(jack, Boy) = {isinstance(jack, Boy)}, isinstance(jack, Person) = {isinstance(jack, Person)}')
    print(f'isinstance(jack_boy, Boy) = {isinstance(jack_boy, Boy)}, isinstance(jack_boy, Person) = {isinstance(jack_boy, Person)}')

    # 特殊变量
    print(f'jack_boy._protected_var = {jack_boy._protected_var}, jack_boy.__private_var = {jack_boy._Boy__private_var}')

    # 结构体
    data = DataFormat(10, 'png')
    print(f'data = {data.id}, {data.name}')

    print('list(range(1, 8, 1)):', list(range(1, 8, 1)))

    data = 'abcdefg'
    for idx in range(len(data) - 1, -1, -1):
        print(data[idx], end=" ")
    print()

    data_iter = Reverse(data)

    print('data_iter:', data_iter)
    print('list(data_iter):', list(data_iter))

    for item in data_iter:
        print(item, end=' ')
    print()

    print('range(5) =', range(5))
