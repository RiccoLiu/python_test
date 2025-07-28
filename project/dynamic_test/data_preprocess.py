

import os
import argparse
from glob import glob

from loguru import logger
import shutil
from shutil import copyfile

def main(input_path, output_path, args):
    for video in os.listdir(input_path):
        pred_results_path = os.path.join(input_path, video)
        pred_files = sorted(glob(pred_results_path + "/*.txt"), key=lambda x: int(x.split("_")[-1].split(".")[0]))
        # logger.info(f"results: {results}")

        output = os.path.join(output_path, video)
        os.makedirs(output, exist_ok=True)
        
        for file in pred_files:
            
            logger.info(f"os.path.basename(file): {os.path.basename(file)}")
            name = os.path.basename(file)
            name = name.replace(".MP4", "")
            logger.info(f"oname: {name}")
   
            output_file = os.path.join(output, name)
            shutil.copy(file, output_file)
            logger.info(f"copy {file} to {output_file}")
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_path', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/AutoTest_20250213_origin")
    parser.add_argument('--input', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/v7.1.1")
    parser.add_argument('--output', type=str, help='预测结果路径', default="/home/lc/work/sdk/dynamic_test/v7.1.1_processed")
    args = parser.parse_args()
    
    logger.info(args)

    main(args.input, args.output, args)

