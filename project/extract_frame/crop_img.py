#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import argparse

def CropImage(input_path, output_path, height_start):
    print(f'input_path:{input_path}, output_path:{output_path}, height_start:{height_start}')

    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if not file.lower().endswith('.jpg'):
                continue

            file_path = os.path.join(root, file)

            try:
                img = cv2.imread(file_path)
                
                if img is None:
                    print(f"Error reading {file_path}")
                    continue

                height, width, _ = img.shape
                if width != 1920 or height != 1080:
                    print(f"Skipping {file_path} because its size is not 1920x1080 (actual size: {width}x{height})")
                    continue

                crop_img = img[height_start:height_start + 384, 0:1920]
                resized_img = cv2.resize(crop_img, (960, 192), interpolation=cv2.INTER_LINEAR)

                output_file_path = os.path.join(output_path, file_path)
                
                output_dir_path = os.path.dirname(output_file_path)
                # print(f'output_dir_path:{output_dir_path}')
                os.makedirs(output_dir_path, exist_ok=True)
                
                cv2.imwrite(output_file_path, resized_img)
                print(f'Processed: {file_path} -> {output_file_path}')

            except Exception as e:
                print(f'Error prossing {file_path}:{e}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop Image')

    parser.add_argument('-in', dest='input_path',
                        type=str,
                        required=True,
                        metavar='input_path',
                        help='frames')
    parser.add_argument('-out', dest='output_path',
                        type=str,
                        required=True,
                        metavar='output_path',
                        help='./tmp')
    
    parser.add_argument('-hs', dest="height_start",
                        type=int,
                        default=630,
                        metavar='crop_height_start',
                        help='630',
                        )
    
    args = parser.parse_args()
    
    CropImage(args.input_path, args.output_path, args.height_start)
