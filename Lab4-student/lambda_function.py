import boto3
import lightgbm
import json

def get_model():
    bucket= boto3.resource('s3').Bucket('deploy-lgbm-sisi')
    bucket.download_file('model/saved_adult_model.txt','/tmp/test_model.txt')
    model= lightgbm.Booster(model_file='/tmp/test_model.txt')
    return model

def predict(data):
    model = get_model()
    result = model.predict(data)
    return result

def lambda_handler(event, context):
    if isinstance(event['body'], str):
        tt=json.loads(event['body'])
    else:
        tt=event['body']

    result = predict([tt['data']])

    return {
        'statusCode': 200,
        'body': json.dumps(result[0])
    }
