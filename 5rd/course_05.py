#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import unittest

'''
    dir:
    help:
'''

def add(a, b):
    """
    这个函数返回两个数的和。
    示例：
    >>> add(2, 3)
    5
    """
    return a + b

def subtract(a, b):
    """
    返回两个数的差。
    
    >>> subtract(3, 2)
    1
    """
    return a - b

class UnitTest(unittest.TestCase):
    # 准备测试环境
    def setUp(self):
        self.saved_argv = sys.argv

    # 清理测试用例
    def tearDown(self):
        sys.argv = self.saved_argv

    # 测试用例
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(5, 10), 15)  
        self.assertEqual(add(6, 3), 9)

    # 跳过这个测试用例
    @unittest.skip("Skipping this test for now")
    def test_add_skip(self):
        self.assertEqual(add(2, 3), 100)

    # 根据条件决定是否跳过这个测试用例
    @unittest.skipIf(1 == 1, "This condition is always True")
    def test_subtract(self):
        self.assertEqual(subtract(3, 2), 4)

    # 这个测试用例会失败
    @unittest.expectedFailure
    def test_add(self):
        self.assertEqual(add(2, 3), 6)

if __name__ == '__main__':
    import os

    # unittest.main()
    # exit(0)

    '''
        dir:
            列出模块内的所有子模块
        help:
            查看模块的使用手册
    '''
    # help(os.getcwd)    
    # print('dir(os.getcwd) = ', dir(os.getcwd))

    '''
        lines = []
        with open(_in_config, 'r') as f:
            lines = f.readlines() 
        
        for line in lines:            
            line.replace('\n', '') # 去掉行尾的换行符

            cols = line.split('/') # 使用 '/' 作为分割符
        
        strip:
            是一种更通用的方法，会移除字符串两端的所有空白字符，包括空格、制表符(\t)和换行符(\n)
    '''

    print(f'os.path.realpath(__file__): {os.path.realpath(__file__)}')
    print(f'os.path.dirname(os.path.realpath(__file__): {os.path.dirname(os.path.realpath(__file__))}')

    '''
        os:
            os.getcwd:
                获取执行应用程序所在的路径，不一定是文件所在路径, 文件所在路径: os.path.realpath(__file__)

            os.listdir(path):
                列出目录下的所有文件和目录

            os.makedirs(path):
                创建目录            

            os.path.exists():
                查看文件或目录是否存在

            os.path.isdir():
                输入的路径是否是目录

            os.path.isfile():
                输入路径是否是文件

            os.path.join('root', 'first', 'second'):   
                返回 'root/first/second' 目录               

            os.path.basename(file):
                通过文件的路径获取文件名       
    
            os.path.dirname(path | file):
                当前路径的父路径
            
            os.path.splitext:
                根据文件最后一个.文件名和扩展名分离    
            
            os.path.realpath(__file__):
                获取当前文件的绝对路径

        sys:
            sys.path.append(path):
                将路径加入到模块的默认搜索路径
            
            sys.path.append(os.path.dirname(os.path.realpath(__file__)))
                将当前文件所在路径加入模块的搜索路径，可以放在包的 __init__.py 里
    '''

    cur_path = os.getcwd()

    print(f'cur_path: {cur_path}')
    print(f'os.path.basename(cur_path): {os.path.basename(cur_path)}')
    print(f'os.path.dirname(cur_path): {os.path.dirname(cur_path)}')

    os_cmd_path = os.path.join(cur_path, 'os_cmd')

    print(f'os_cmd_path: {os_cmd_path}, cur_path:{cur_path}')    
    print(f'os.getcwd() = {os.getcwd()}')
    print(f'os_cmd_path = {os.path.exists(os_cmd_path)}')
    print(r'os.path.exists(course_05.py):', os.path.exists('course_05.py'))

    for it in os.listdir(os.getcwd()):
        print(f'it: {it}, is_dir: {os.path.isdir(it)}, is_file: {os.path.isfile(it)}')

    '''
        脚本中执行应用程序的几种方式：
            os.system:
                执行简单的系统命令,无法捕获系统的输出,不方便进行重定向和管道操作, 安全性差
            subpross:
                可以处理子进程的标准输出,错误和返回值, 通过管道或文件来与子进程交互, 安全性高
            shutil:
                主要对文件和目录的操作,比如:复制、删除、移动等  
                    shutil.copy(src, dst)       复制文件
                    shutil.copy2(src, dst)	    复制文件，包含元数据
                    shutil.copytree(src, dst)	递归复制整个目录树

                    shutil.remove(path)	        删除文件
                    shutil.rmtree(path)	        递归删除整个目录树

                    shutil.move(src, dst)	    移动文件或目录
                    
                    shutil.make_archive()	    创建压缩文件
                    shutil.unpack_archive()	    解压文件

                    shutil.link()	            创建硬链接
                    shutil.symlink()	        创建符号链接（软链接）

                    shutil.disk_usage(path)	    获取磁盘使用情况
                    shutil.chown()	            改变文件或目录的所有者和所属组
                    shutil.copymode()	        复制文件权限模式
                    shutil.copystat()	        复制文件的状态信息
    '''

    src_file = 'Open Source Software Notice.txt'
    dst = 'os_cmd'
    
    if not os.path.exists(dst):
        os.system(f'mkdir -p {dst}' )

    # 需要把字符串的空格前插入转义字符 \ -> 'Open\ Source\ Software\ Notice.txt'

    # src_file = src_file.replace(' ', '\\ ')  # 空格替换为 "\ " 
    # src_file = ''.join(['\\ ' if c == ' ' else c for c in src_file]) # 遍历字符, 如果是空格改为 "\ "

    if 0 != os.system(f'cp "{src_file}" "{dst}"'):
        print(f'os system(cp "{src_file}" "{dst}") faild')
        sys.exit(-1)

    import subprocess as sp
    
    src_file = src_file.replace('\\', '')

    dst = "sp_cmd"

    if not os.path.exists(dst):
        os.system(f'mkdir -p {dst}' )

    print(f"src_file = {src_file}, dst = {dst}")
    
    cmd = f'cp {src_file} {dst}'
    cmd2 = f'cp "{src_file}" "{dst}"'

    st = sp.run(f'cp "{src_file}" "{dst}"', shell=True)
    if 0 != st.returncode:
        print(f'cp "{src_file}" "{dst}" failed')
        sys.exit(-1)

    import shutil

    dst = "st_cmd"

    if not os.path.exists(dst):
        os.system(f'mkdir -p {dst}' )

    shutil.copy(f"{src_file}", dst)
    # shutil.copytree(dst, dst + '_copy')

    '''
        glob 
            通过模式匹配查找目录或文件

            glob.glob(pattern)	返回匹配模式的文件路径列表。
            glob.iglob(pattern)	与 glob() 类似，但返回的是一个生成器（适用于大文件）。
            glob.glob('**/pattern', recursive=True)	递归查找子目录中的文件（** 是递归匹配）。
    '''

    import glob
    # files = glob.glob('/home/lc/work/slam_oncloud/output/2024060103shanghai3-test/*.geojson')

    file_root = '/home/lc/work/sdk/slam_oncloud/output/2024060103shanghai3-2023092601shenyang22'

    files = glob.glob(file_root + '/semantic/**/*geojson', recursive=True)
    print('semantic files:', files, 'size:', len(files))

    files = glob.glob(file_root + '/semantic_onboard/**/*geojson', recursive=True)
    print('semantic_onboard files:', files, 'size:', len(files))

    files = glob.glob(file_root + '/trajectory/**/*geojson', recursive=True)
    print('trajectory files:', files, 'size:', len(files))

    print('-----------------------------')

    '''
        argparse
            解析命令行参数
    '''
    import argparse

    parser = argparse.ArgumentParser()


    # 位置参数 - 必须传入的
    parser.add_argument("input_path", type=str, help="输入数据路径.")
    parser.add_argument("output_path", type=str, help="输出数据路径.")

    # 关键字参数 - required 的参数必须通过关键字传入
    parser.add_argument("-m", "--module", type=str, dest="module_name", metavar="module", help="指定要输出模块范围")
    parser.add_argument("-t", "--tile", type=int, dest="tile_name", metavar="tile_id", required=True, help="指定要输出的tile块")
    
    # 默认参数, -f 就为 True, 没有 -f 就是 False, 
    parser.add_argument("-f", "--flag", action="store_true", help="启用详细模式")
    parser.add_argument("-n", "--name", type=str, default="default.txt", metavar="file name", help="输出文件名")

    args = parser.parse_args()

    for k, v in sorted(vars(args).items()):
        print(k, '=', v, flush=True)

    # print(f'input_path:{args.input_path}, output_path:{args.output_path}, module_name:{args.module_name}, tile_name = {args.tile_name}, flag = {args.flag}, name = {args.name}')

    # ./course_05.py i_path o_path -t 10 -f
    # input_path:i_path, output_path:o_path, module_name:None, tile_name = 10, flag = True, name = default.txt
    print('-----------------------------') 

    '''
        数学
            math:
                math.pi
                math.sin
                math.cos
            random:
                random.choice
                random.sample
                random.random
            statistics:
    '''
    import math
    print('math.pi:', math.pi)
    print('math.cos(math.pi / 3):', math.cos(math.pi / 3))
    print('math.sin(math.pi / 6):', math.sin(math.pi / 6))
    print('math.log(27, 3):', math.log(27, 3))

    import random

    random_choice = random.choice([1, 'as', 3, 'bd', 6])
    random_sample = random.sample(range(100), 10)
    random_val = random.random()

    print('random_choice:', random_choice)
    print('random_sample:', random_sample)
    print('random_val:', random_val)

    import statistics
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    statistics.mean(data)

    print(f'data mean: {statistics.mean(data)}, median: {statistics.median(data)}, variance: {statistics.variance(data)}')

    print('-----------------------------') 

    '''
        datatime: 日期时间
            年: %y(25)  %Y(2025)
            月: %m(01)  %b(Jan) %B(January)
            日: %d(03)
            星期: %A(Friday)

            时: %H
            分: %M
            秒: %S
    '''

    import datetime
    from datetime import date

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(datetime.datetime.now().strftime("%d %b %Y is a %A on the %d day of %B."))

    print('-----------------------------') 

    '''
        timeit:
            性能测试
            Timer:
                # 计算 code_to_test耗时,可以是函数名,也可以是字符串,setup_code 执行代码前运行的代码
                timer = Timer('code_to_test', 'setup_code') 
                costtime = timer.timeit(number=1000)
            timeit:
                costtime = timeit.timeit('sum(range(1000))', number=1000)
    '''
    
    from timeit import Timer

    timer = Timer('sum(range(1000))')
    execution_time = timer.timeit(number=1000)  # 执行 1000 次 

    print(f"sum(range(1000)) cost time: {execution_time} seconds")

    print('-----------------------------') 

    '''
        doctest & unittest:
            doctest:
                用于提取文档字符串的代码，验证输入输出是否正确   

                doctest.testmod()
                    默认测试当前模块的所有文档字符串，确认输入输出是否正确

                    doctest.run_docstring_examples(add, globals())
                        测试当前模块中的 add 的文档字符串，确认输入输出是否正常
                                    
                doctest.testfile()
                    测试外部文档字符串文件，
                    # example.txt
                    >>> add(2, 3)
                    5
                    >>> add(0, 0)
                    0    
                doctest.testfile('example.txt')
    
            unittest:
                用于单元测试
                
                # 测试用例
                unittest.TestCase:
                    def setUp(self):    # 准备单元测试环境
                    def tearDown(self): # 清理但远测试环境S

                    使用装饰器修改期望测试用例的结果
                        @unittest.skip("Skipping this test for now") # 跳过此测试用例
                        @unittest.skipIf(1 == 1, "This condition is always True") # 根据条件跳过测试用例
                        @unittest.expectedFailure   # 期望测试用例会失败
                    
                unittest.main() # 执行所有测试用例   
    '''

    import doctest
    doctest.testmod() # 测试 add(2, 3) 输出是否为 5, subtract(3, 2) 输出是否为 1

    doctest.run_docstring_examples(add, globals())      # add 文档字符串测试
    doctest.run_docstring_examples(subtract, globals()) # subtract 文档字符创测试

    # 使用 unittest 与 argparse 会有冲突
    print('-----------------------------') 
