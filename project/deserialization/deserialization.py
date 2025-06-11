#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from proto import DataCollection_pb2

if __name__ == '__main__':
    # with open('data/identify/1717236384630.record', 'rb') as f:
    #     identify_data = f.read()

    # dc_info = DataCollection_pb2.DataCollectInfo()
    # dc_info.ParseFromString(identify_data)

    # print('---- data_type: ----')
    # print(dc_info.data_type)

    # print('---- common_data: ----')
    # print(dc_info.common_data)

    # identify_info = DataCollection_pb2.IdentifyResults()
    # identify_info.ParseFromString(dc_info.collect_data)

    # print('---- IdentifyResults: ----')
    # print(identify_info)

    # with open('data/trajectory/1717229560230.record', 'rb') as f:
    #     trajectory_data = f.read()

    # dc_info = DataCollection_pb2.DataCollectInfo()
    # dc_info.ParseFromString(trajectory_data)

    # print('---- data_type: ----')
    # print(dc_info.data_type)

    # print('---- common_data: ----')
    # print(dc_info.common_data)

    # location_info = DataCollection_pb2.LocationData()
    # location_info.ParseFromString(dc_info.collect_data)

    # print('---- location_info: ----')
    # print(location_info)
        
    # with open('dc/semantic/1719027076.record', 'rb') as f:
    #     data = f.read()
    
    # dc_info = DataCollection_pb2.DataCollectInfo()
    # dc_info.ParseFromString(data)
    
    # print('---- data_type: ----')
    # print(dc_info.data_type)
    
    # sem_info = DataCollection_pb2.SemanticMapList()
    # sem_info.ParseFromString(dc_info.collect_data)
    
    # print('---- sem_info: ----')
    # print(sem_info)    

    # with open('dc/identify/-4601385357183.record', 'rb') as f:
    #     data = f.read()
    
    # dc_info = DataCollection_pb2.DataCollectInfo()
    # dc_info.ParseFromString(data)
    
    # print('---- data_type: ----')
    # print(dc_info.data_type)
    
    # iden_info = DataCollection_pb2.IdentifyResults()
    # iden_info.ParseFromString(dc_info.collect_data)
    
    # print('---- iden_info: ----')
    # print(iden_info)

    # base_dir = 'LSJA24U91L/semantic'
    
    # for file in os.listdir(base_dir):
    #     if not file.endswith('.record'):
    #         continue
        
    #     with open(os.path.join(base_dir, file), 'rb') as f:
    #         data = f.read()
        
    #     dc_info = DataCollection_pb2.DataCollectInfo()
    #     dc_info.ParseFromString(data)
    
    #     print(f'---- file: {os.path.join(base_dir, file)} ----')
        
    #     print('---- data_type: ----')
    #     print(dc_info.data_type)

    #     sem_info = DataCollection_pb2.SemanticMapList()
    #     sem_info.ParseFromString(dc_info.collect_data)

    #     print('---- sem_info: ----')
    #     print(sem_info) 

    base_dir = 'LSJA24U91L/semantic'
    
    for file in os.listdir(base_dir):
        if not file.endswith('.record'):
            continue
        
        with open(os.path.join(base_dir, file), 'rb') as f:
            data = f.read()
        
        dc_info = DataCollection_pb2.DataCollectInfo()
        dc_info.ParseFromString(data)
    
        print(f'---- file: {os.path.join(base_dir, file)} ----')
        
        print('---- data_type: ----')
        print(dc_info.data_type)

        sem_info = DataCollection_pb2.SemanticMapList()
        sem_info.ParseFromString(dc_info.collect_data)

        print('---- sem_info: ----')
        print(sem_info) 


    base_dir = 'LSJA24U91L/trajectory'
    
    for file in os.listdir(base_dir):
        if not file.endswith('.record'):
            continue
        
        with open(os.path.join(base_dir, file), 'rb') as f:
            data = f.read()
        
        dc_info = DataCollection_pb2.DataCollectInfo()
        dc_info.ParseFromString(data)
    
        print(f'---- file: {os.path.join(base_dir, file)} ----')
        
        print('---- data_type: ----')
        print(dc_info.data_type)

        location_info = DataCollection_pb2.LocationData()
        location_info.ParseFromString(dc_info.collect_data)

        print('---- location_info: ----')
        print(location_info) 

    # base_dir = 'LSJA24U91L/identify'

    # for file in os.listdir(base_dir):
    #     if not file.endswith('.record'):
    #         continue
        
    #     with open(os.path.join(base_dir, file), 'rb') as f:
    #         data = f.read()
        
    #     dc_info = DataCollection_pb2.DataCollectInfo()
    #     dc_info.ParseFromString(data)
    
    #     print(f'---- file: {os.path.join(base_dir, file)} ----')
        
    #     print('---- data_type: ----')
    #     print(dc_info.data_type)

    #     iden_info = DataCollection_pb2.IdentifyResults()
    #     iden_info.ParseFromString(dc_info.collect_data)

    #     print('---- identify_info: ----')
    #     print(location_info) 



