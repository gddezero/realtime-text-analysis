from __future__ import print_function

import base64
import json
import boto3
import time

BATCH_SIZE = 25
ENTITY_STREAM_NAME = 'entity-stream'

print('Loading function')
client_comprehend = boto3.client(service_name='comprehend')
client_firehose = boto3.client(service_name='firehose')

def lambda_handler(event, context):
    output = []
    text_list = []
    payload_list = []
    recordId_list = []
    title_list = []
    i = 0
    num_record = len(event['records'])

    for record in event['records']:
        i += 1
        print(record['recordId'])
        payload = json.loads(base64.b64decode(record['data']))

        payload_list.append(payload)
        text_list.append(payload['text'])
        recordId_list.append(record['recordId'])
        title_list.append(payload['title'])
        
        # batch detect
        if (i % BATCH_SIZE == 0 or i >= num_record):
            print('processing batch #{}'.format(i / BATCH_SIZE))
            
            # detect key phrases
            key_phrase_list = get_key_phrase_list(text_list)
        
            # detect sentiments
            sentiment_list = get_sentiment_list(text_list)
            
            # detect named entities and send to another firehose
            entity_list = send_entity(text_list, sentiment_list, recordId_list, title_list)

            # save text analysis result to payload
            j = 0
            for payload in payload_list:
                payload['@timestamp'] = str(int(time.time() * 1000))
                payload['key_phrase'] = key_phrase_list[j]
                payload['sentiment'] = sentiment_list[j]
                payload['entity'] = {
                    'text': entity_list[j][0],
                    'type': entity_list[j][1]
                }
                payload['doc_type'] = 'doc'
                
                output_record = {
                    'recordId': recordId_list[j],
                    'result': 'Ok',
                    'data': base64.b64encode(json.dumps(payload))
                }
                output.append(output_record)
                
                j += 1
        
            text_list = []
            payload_list = []
            recordId_list = []

    print('Successfully processed {} records.'.format(num_record))

    return {'records': output}
    
def send_entity(text_list, sentiment_list, recordId_list, title_list):
    response = client_comprehend.batch_detect_entities(TextList=text_list, LanguageCode='en')
    entity_list = []
    
    for result, sentiment, recordId, title in\
        zip(response['ResultList'], sentiment_list, recordId_list, title_list):
        #entities = list(set((x['Text'], x['Type']) for x in result['Entities'] if x['Type'] != 'QUANTITY'))

        entity_text = []
        entity_type = []
        for entity in result['Entities']:
            payload = {
                '@timestamp': str(int(time.time() * 1000)),
                'entity': {
                    'text': entity['Text'],
                    'type': entity['Type']
                    
                },
                'doc_type': 'entity',
                'sentiment': sentiment,
                'parent': recordId[:-6],
                'title': title
                
            }
            client_firehose.put_record(DeliveryStreamName=ENTITY_STREAM_NAME, Record={'Data':json.dumps(payload)})
            entity_text.append(entity['Text'])
            entity_type.append(entity['Type'])

        entity_list.append((entity_text, entity_type))
        
    return entity_list
    
def get_key_phrase_list(text_list):
    response = client_comprehend.batch_detect_key_phrases(TextList=text_list, LanguageCode='en')
    key_phrase_list = []

    for result in response['ResultList']:
        key_phrases = list(set(x['Text'] for x in result['KeyPhrases']))
        key_phrase_list.append(key_phrases)
    return key_phrase_list
    
def get_sentiment_list(text_list):
    response = client_comprehend.batch_detect_sentiment(TextList=text_list, LanguageCode='en')
    sentiment_list = []

    for result in response['ResultList']:
        sentiment = result['Sentiment']
        sentiment_list.append(sentiment)
    return sentiment_list