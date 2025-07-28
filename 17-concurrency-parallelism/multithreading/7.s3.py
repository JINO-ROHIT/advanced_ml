"""In this tutorial, we will see how to upload multiple files to an s3 bucket using multi threading"""

import threading
import concurrent.futures

import boto3
import os

from dotenv import load_dotenv
load_dotenv()

session = boto3.Session(
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key = os.getenv('AWS_SECRET_KEY')
)

s3 = session.resource('s3')

def upload_to_s3(path):
    s3.meta.client.upload_file(Filename = f"{path}",
                        Bucket = os.getenv('AWS_BUCKET_NAME'), 
                        Key = f"subfolder/{path}",
                        )
    print(f"thread {threading.get_native_id()} uploaded {path}")

files_paths = []
for root, _, files in os.walk('data'):
    for file in files:
        files_paths.append(os.path.join(root, file).replace('\\', '/')) # windows path

with concurrent.futures.ThreadPoolExecutor(max_workers = 3) as executor:
    executor.map(upload_to_s3, files_paths)