import boto3

def get_boto_session(region_name='us-east-1', *args, **kwargs):
    return boto3.Session(region_name=region_name, *args, **kwargs)