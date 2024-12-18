#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

def rename_bin2record(dir):
    for filename in os.listdir(dir):
        # 检查文件是否以 .bin 结尾
        if filename.endswith('.bin'):
            # 构造新的文件名
            new_filename = filename.replace('.bin', '.record')
            
            # 获取原文件的完整路径
            old_file = os.path.join(dir, filename)
            
            # 获取新文件的完整路径
            new_file = os.path.join(dir, new_filename)
            
            # 重命名文件
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == '__main__':
    rename_bin2record(os.getcwd())