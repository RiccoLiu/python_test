
### 编译 .proto 文件
protoc --proto_path=proto --python_out=proto proto/CameraParam.proto proto/DataCollection.proto proto/SLAMCommon.proto

### 修改导入的路径

```
# import CameraParam_pb2 as CameraParam__pb2
# import SLAMCommon_pb2 as SLAMCommon__pb2

from proto import CameraParam_pb2 as CameraParam__pb2
from proto import SLAMCommon_pb2 as SLAMCommon__pb2
```