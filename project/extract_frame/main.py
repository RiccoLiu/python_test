#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import argparse
import traceback
import shutil

from multiprocessing import Pool, cpu_count
from minio_api import MinioAPI as minio

remote_server = '192.168.2.7'

remote_server_path = [
'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/常规路测',
'BugInfo-Ⅰ（20240201启用）/CP121010.VW(CNS3.0-GP)/常规路测',

'BugInfo-Ⅰ（20240201启用）/ARS2',
'BugInfo-Ⅰ（20240730启用）/CP121019.南北大众_GP_SOP2/常规路测',

'BugInfo(2024.09.05启用)/CP121019.南北大众_GP_SOP2/常规路测',
]

def SMBDownload(cur_path, remote_path, ouput_path):
    print('******************************************')
    print(f'SMBDownload remote_path:{remote_path}, ouput_path:{ouput_path}')

    local_path = os.path.join(ouput_path, remote_path)
    if os.path.exists(local_path):
        print(f'local_path already exist, skip the {local_path}')
        return

    os.makedirs(local_path, exist_ok=True)
    os.chdir(local_path)

    cols = remote_path.split('/')
    
    shared_path = cols[0]
    target_path = os.path.join(*cols[1:])  # 将列表转换为路径

    # timeout 86400;
    cmd = f'smbclient "//192.168.2.7/{shared_path}" -U mansion%MXzaq1 -c "timeout 86400; recurse ON; prompt OFF; cd {target_path}; mget *" '
    if 0 != os.system(cmd):
        print(f'run cmd faild, cmd = ', cmd)
        
        with open('failed.txt', 'a+') as f:
            f.write(f'remote_path:{remote_path}\n')
            f.write(f'local_path:{local_path}\n')
            f.write(f'run cmd faild, cmd = {cmd}\n')
        return

    os.chdir(cur_path)

def ExtractVideoFrame(video_root_path, video_path, output_path):
    print('******************************************')
    print(f'ExtractVideoFrame video_path:{video_path}, output_path:{output_path}')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, _, files in os.walk(video_path):
        for file in files:
            video_file = os.path.join(root, file)
            relative_path = os.path.relpath(video_file, video_root_path)

            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_name = os.path.splitext(relative_path)[0]
            video_output_path = os.path.join(output_path, video_name)
            
            if not os.path.exists(video_output_path):
                os.makedirs(video_output_path)

            print('-----------------------------------')
            print(f'ExtractVideoFrame video_file:{video_file}')
            print(f'ExtractVideoFrame output_path:{video_output_path}')

            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Failed to open video: {video_file}")
                continue

            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save one frame per second
                if frame_count % frame_rate == 0:
                    output_file = os.path.join(video_output_path, f"frame_{saved_count:08d}.jpg")
                    cv2.imwrite(output_file, frame)
                    saved_count += 1

                frame_count += 1

            cap.release()   

            print(f"ExtractVideoFrame Extracted {saved_count} frames ")
            print('-----------------------------------')
            

def Upload(local_path, remote_path):
    print('******************************************')
    print(f'Upload local_path:{local_path}, remote_path:{remote_path}')

    minio.upload_objects(local_path, remote_path)

def RemoveLocalData(video_path, frame_path):
    if os.path.exists(video_path):
        try:
            shutil.rmtree(video_path)
            print(f"目录 '{video_path}' 及其内容已成功删除")
        except Exception as e:
            print(f"删除失败: {e}")
    else:
        print(f"目录 '{video_path}' 不存在")

def ProcessOne(cur_path, remote_path):
    root_path = 'lc'
    
    video_root_path = os.path.join(root_path, 'origin_videos')
    frame_root_path = os.path.join(root_path, 'extracted')

    SMBDownload(cur_path, remote_path, video_root_path)

    video_path = os.path.join(video_root_path, remote_path)
    if not os.path.exists(video_path):
        print(f'video_path is not exists, skip {video_path}')
        return

    ExtractVideoFrame(video_root_path, video_path, frame_root_path)

    frame_path = os.path.join(frame_root_path, remote_path)
    if not os.path.exists(frame_path):
        print(f'frame_path is not exists, skip {frame_path}')
        return 
    
    Upload(frame_path, f'perceptiondata40/{frame_path}') 
    
    # RemoveLocalData(video_root_path, frame_root_path)

'''
    lc/
      - origin_videos/
      - extracted/
      - croped/
'''

if __name__ == '__main__':
    result = []
    try:
        pool = Pool(1)
        for remote_path in remote_server_path:
            result.append((remote_path, pool.apply_async(ProcessOne, args=(os.getcwd(), remote_path))))
        pool.close()
        pool.join()

        for ret in result:
            print(f"task:{ret[0]}, ret:{ret[1].get()}")
    except:
        traceback.print_exc()
