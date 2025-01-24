import platform
import minio
import os
import zipfile
from minio import error
from concurrent.futures import ThreadPoolExecutor
import time
import certifi
import urllib3
import sys

max_workers = 50

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class MinioAPI(object):
    """
    客户端
    """

    def __init__(self):
        timeout = 300

        http_client = urllib3.PoolManager(
            timeout=urllib3.util.Timeout(connect=timeout, read=timeout),
            maxsize=max_workers,
            cert_reqs='CERT_REQUIRED',
            ca_certs=os.environ.get('SSL_CERT_FILE') or certifi.where(),
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.__client = minio.Minio(
            endpoint=os.getenv('URL'),
            access_key=os.getenv('ACCESS_ID'),
            secret_key=os.getenv('ACCESS_KEY'),
            secure=False, http_client=http_client)

    @staticmethod
    def split_bucket_path(path):
        bucket = path.split('/')[0]
        obj_list = path.split('/')[1:]
        key_path = '/'.join(obj_list)
        return bucket, key_path

    @staticmethod
    def extract_all_zip_files(directory, is_rm_zip=True):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.zip'):
                    zip_file_path = os.path.join(root, file)
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    print(f"已解压: {zip_file_path}")
                    if is_rm_zip:
                        os.remove(zip_file_path)

    def list_objects(self, bucket_name, prefix):
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        for i in range(5):
            try:
                objects = self.__client.list_objects(bucket_name, prefix=prefix, recursive=True)
                return [obj.object_name for obj in objects if not obj.object_name.endswith('/')]
            except error.S3Error as exception:
                with open('minio_log.txt', 'a') as f:
                    f.write(f'{bucket_name}/{prefix} 第{i}次list_objects失败 异常：{exception}\n')

        with open('minio_log.txt', 'a') as f:
            f.write(f'{bucket_name}/{prefix} list_objects失败\n')
        print(f'{bucket_name}/{prefix} list_objects失败')
        sys.exit(-1)

    def check_dir_exist(self, s3_path):
        bucket_name, prefix = self.split_bucket_path(s3_path)
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        objects = list(self.__client.list_objects(bucket_name, prefix=prefix, recursive=False))
        if len(objects) > 0:
            print(f'{s3_path} exist.')
            return True
        else:
            print(f'{s3_path} not exist!!!')
            return False

    def download_file(self, bucket: str, key: str, file_name: str, num_retries=5):
        for i in range(num_retries):
            try:
                data = self.__client.fget_object(bucket, key, file_name)
                if data:
                    # print(f'{bucket}/{key} 下载完成')
                    return True
            except error.S3Error as exception:
                with open('minio_log.txt', 'a') as f:
                    f.write(f'{bucket}/{key} 第{i}次下载失败 异常：{exception}\n')

        with open('minio_log.txt', 'a') as f:
            f.write(f'{bucket}/{key} 下载失败\n')
        print(f'{bucket}/{key} 下载失败')
        sys.exit(-1)

    def download_objects(self, s3_path: str, file_path: str, is_file=False):
        print(f"minio_path:{s3_path} -> local_path:{file_path}")
        start = time.time()
        bucket, key_path = self.split_bucket_path(s3_path)

        if is_file:
            object_names = [key_path]
        else:
            object_names = self.list_objects(bucket, key_path)

        if len(object_names) == 0:
            with open('minio_log.txt', 'a') as f:
                f.write(f'minio {s3_path} is empty!!!\n')
            return False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            f = []
            for object_name in object_names:
                if not is_file:
                    if not key_path.endswith('/'):
                        key_path = key_path + '/'
                    split_str = object_name.replace(key_path, "")
                    local_path = file_path + '/' + split_str
                else:
                    local_path = os.path.join(file_path, os.path.basename(object_name))
                if platform.system() == 'Windows':
                    local_path = local_path.replace("/", "\\")
                f.append(executor.submit(self.download_file, bucket, object_name, local_path))
            for sub in f:
                sub.result()
        end = time.time()
        print(f'{s3_path} 下载耗时: {end - start} 秒')

    def upload_file(self, file_path: str, bucket: str, file_name: str, key, num_retries=5):
        for i in range(num_retries):
            try:
                status = self.__client.fput_object(bucket, key + '/' + file_name, file_path,
                                                   content_type='application/json')
                if status:
                    # print(f'{file_path} 上传完成')
                    return True
            except error.S3Error as exception:
                with open('minio_log.txt', 'a') as f:
                    f.write(f'{file_path} 第{i}次上传失败 异常：{exception}\n')

        with open('minio_log.txt', 'a') as f:
            f.write(f'{file_path} 上传失败\n')
        print(f'{key}/{file_name} 上传失败')
        sys.exit(-1)

    def upload_objects(self, local_path: str, s3_path: str, is_nest=True):
        print(f"local_path:{local_path} -> minio_path:{s3_path}")
        start = time.time()
        # print('开始上传文件')
        bucket, key_path = self.split_bucket_path(s3_path)
        if os.path.isdir(local_path):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                f = []
                for path, dirs, file_lst in os.walk(local_path):
                    for file_name in file_lst:
                        current_path = os.path.join(path, file_name)
                        if not is_nest:
                            s3_name = file_name
                        else:
                            s3_name = current_path.replace(local_path, "")[1:]
                            if platform.system() == 'Windows':
                                s3_name = s3_name.replace("\\", "/")
                        f.append(executor.submit(self.upload_file, current_path, bucket, s3_name, key_path))
                for sub in f:
                    sub.result()
        else:
            file_name = os.path.basename(local_path)
            self.upload_file(local_path, bucket, file_name, key_path)

        end = time.time()
        print(f'上传耗时: {end - start} 秒')
