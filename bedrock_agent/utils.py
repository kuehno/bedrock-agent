import boto3
import json

def get_boto_session(region_name='us-east-1', *args, **kwargs):
    return boto3.Session(region_name=region_name, *args, **kwargs)

def get_embedding(input_text: str, session: boto3.Session = None, model_id: str = "amazon.titan-embed-text-v2:0", region_name: str = "us-east-1", dimensions: int = 256):
    if not session:
        session = get_boto_session(region_name=region_name)
        
    client = session.client("bedrock-runtime", region_name=region_name)
    
    native_request = {"inputText": input_text, "dimensions": dimensions}
    request = json.dumps(native_request)
    
    response = client.invoke_model(
        modelId=model_id, 
        body=request
        )
    
    model_response = json.loads(response["body"].read())
    embedding = model_response["embedding"]
    return embedding