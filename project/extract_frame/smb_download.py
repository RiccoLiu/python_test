#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import argparse
from multiprocessing import Pool, cpu_count
import traceback

remote_server = '192.168.2.7'

# remote_server_path = [
# 'BugInfo(2023.12.7启用)/CP121008.VW(GP-AR-NAVIGATION)/沈阳路测',
# # 'BugInfo(2023.12.7启用)/CP121008.VW(GP-AR-NAVIGATION)/内部bug相关附件/VisionSDK bug相关附件',

# # 'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/常规路测',
# 'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/上海路测',
# 'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/外部bug相关附件',
# 'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/验收路测',

# 'BugInfo-Ⅰ（20240201启用）/CP121008.VW(GP-AR-NAVIGATION)/客户路测DVR原视频',
# # 'BugInfo-Ⅰ（20240201启用）/CP121010.VW(CNS3.0-GP)/常规路测',
# 'BugInfo-Ⅰ（20240201启用）/CP121010.VW(CNS3.0-GP)/客户路测',
# 'BugInfo-Ⅰ（20240201启用）/CP121019.南北大众_GP_SOP2/常规路测',
# 'BugInfo-Ⅰ（20240201启用）/CP121019.南北大众_GP_SOP2/客户路测',
# # 'BugInfo-Ⅰ（20240201启用）/ARS2',

# # 'BugInfo-Ⅰ（20240730启用）/CP121019.南北大众_GP_SOP2/常规路测',
# 'BugInfo-Ⅰ（20240730启用）/CP121019.南北大众_GP_SOP2/客户路测',

# 'BugInfo(2024.09.05启用)/CP121017.VW_CNS3.0_GP_SOP3/常规路测',
# # 'BugInfo(2024.09.05启用)/CP121019.南北大众_GP_SOP2/常规路测',
# 'BugInfo(2024.09.05启用)/CP121019.南北大众_GP_SOP2/客户路测',

# # BugInfo(2024.09.05启用)/CP121019.南北大众_GP_SOP2/常规路测/failed.txt
# # BugInfo-Ⅰ（20240730启用）/CP121019.南北大众_GP_SOP2/常规路测/failed.txt
# # BugInfo-Ⅰ（20240201启用）/ARS2/failed.txt
# # BugInfo-Ⅰ（20240201启用）/CP121010.VW(CNS3.0-GP)/常规路测/failed.txt
# # BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/常规路测/failed.txt
# ]

remote_server_path = [
'BugInfo(2023.12.7启用)/CP121010.VW(CNS3.0-GP)/常规路测', # done
'BugInfo-Ⅰ（20240201启用）/CP121010.VW(CNS3.0-GP)/常规路测', # done

'BugInfo-Ⅰ（20240201启用）/ARS2', # done
'BugInfo-Ⅰ（20240730启用）/CP121019.南北大众_GP_SOP2/常规路测', # done

'BugInfo(2024.09.05启用)/CP121019.南北大众_GP_SOP2/常规路测',
]

def DownloadDirection(cur_path, remote_path, ouput_path):
    os.chdir(cur_path)
    
    local_path = os.path.join(ouput_path, remote_path)
    if os.path.exists(local_path):
        print(f'{local_path} already exist..')
        return

    os.makedirs(local_path, exist_ok=True)
    os.chdir(local_path)

    cols = remote_path.split('/')
    
    shared_path = cols[0]
    target_path = os.path.join(*cols[1:])  # 将列表转换为路径

    print(f'------------------------')
    print(f'cur_path:{cur_path}')
    print(f'remote_path:{remote_path}')
    print(f'local_path:{local_path}')
    print(f'shared_path:{shared_path}')
    print(f'target_path:{target_path}')

    # timeout 86400; 
    cmd = f'smbclient "//192.168.2.7/{shared_path}" -U mansion%MXzaq1 -c "timeout 86400; recurse ON; prompt OFF; cd {target_path}; mget *" '
    print(f'cmd:{cmd}')
    
    if 0 != os.system(cmd):
        print(f'run cmd faild, cmd = ', cmd)
        
        with open('failed.txt', 'a+') as f:
            f.write(f'remote_path:{remote_path}\n')
            f.write(f'local_path:{local_path}\n')
            f.write(f'shared_path:{shared_path}\n')
            f.write(f'target_path:{target_path}\n')
            f.write(f'run cmd faild, cmd = {cmd}\n')

    print(f'------------------------')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SMB Download Remote Directory')
    parser.add_argument('-out', dest='output_path',
                        type=str,
                        metavar='output_path',
                        help='./tmp')

    args = parser.parse_args()

    output_path = args.output_path

    result = []
    try:
        pool = Pool(8)
        for remote_path in remote_server_path:
            result.append((remote_path, pool.apply_async(DownloadDirection, args=(os.getcwd(), remote_path, output_path))))
        pool.close()
        pool.join()

        for ret in result:
            print(f"task:{ret[0]}, ret:{ret[1].get()}")
    except:
        traceback.print_exc()
