#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import cv2
import argparse

def ExtractVideoFrame(video_path, output_path):
    """
    Extracts one frame per second from all video files in the given directory and saves them as JPG images.

    :param video_path: Path to the directory containing video files.
    :param output_path: Path to the directory to save extracted frames.
    """

    print(f'video_path = {video_path}, output_path = {output_path}')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, _, files in os.walk(video_path):
        for file in files:
            video_file = os.path.join(root, file)

            # Check if the file is a video (can be extended for more formats)
            if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            # Create a subdirectory for each video file's frames
            video_name = os.path.splitext(video_file)[0]
            video_output_path = os.path.join(output_path, video_name)
            
            if not os.path.exists(video_output_path):
                os.makedirs(video_output_path)

            print('-----------------------------------')
            print(f'output_path:{output_path}')
            print(f'root:{root}')
            print(f'video_name:{video_name}')

            print(f'video_file:{video_file}')
            print(f'video_output_path:{video_output_path}')
            
            print('-----------------------------------')

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
            print(f"Extracted {saved_count} frames from {video_file}.")

'''
./extract_frame.py -in 'BugInfo(2023.12.7启用)' -out ./tmp
'''

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Extract Video Frame')

    parser.add_argument('-in', dest='video_path',
                        type=str,
                        required=True,
                        metavar='video_path',
                        help='BugInfo(2023.12.7启用)')
    parser.add_argument('-out', dest='output_path',
                        type=str,
                        metavar='output_path',
                        help='./tmp')

    args = parser.parse_args()

    video_path = args.video_path
    frame_path = args.output_path
    
    if frame_path == None:
        frame_path = os.path.join(video_path, "frames")

    ExtractVideoFrame(video_path, frame_path)

