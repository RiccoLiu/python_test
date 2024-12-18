#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from proto import DataCollection_pb2

if __name__ == '__main__':
    with open('data/identify/1717236384630.record', 'rb') as f:
        identify_data = f.read()

    dc_info = DataCollection_pb2.DataCollectInfo()
    dc_info.ParseFromString(identify_data)

    print('---- data_type: ----')
    print(dc_info.data_type)

    print('---- common_data: ----')
    print(dc_info.common_data)

    identify_info = DataCollection_pb2.IdentifyResults()
    identify_info.ParseFromString(dc_info.collect_data)

    print('---- IdentifyResults: ----')
    print(identify_info)

    with open('data/trajectory/1717229560230.record', 'rb') as f:
        trajectory_data = f.read()

    dc_info = DataCollection_pb2.DataCollectInfo()
    dc_info.ParseFromString(trajectory_data)

    print('---- data_type: ----')
    print(dc_info.data_type)

    print('---- common_data: ----')
    print(dc_info.common_data)

    location_info = DataCollection_pb2.LocationData()
    location_info.ParseFromString(dc_info.collect_data)

    print('---- location_info: ----')
    print(location_info)