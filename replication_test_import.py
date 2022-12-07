import sys
import gc
import os
import time
import json
import weaviate


# Variables
WEAVIATE_URL    = 'http:///'
BATCH_SIZE      = 100
SPHERE_DATASET  = '../sphere.100M.jsonl'


def prepare_client():
    client = weaviate.Client(
        url=WEAVIATE_URL,
        timeout_config=120,
    )

    client.batch.configure(
        batch_size=BATCH_SIZE, 
        dynamic=True,
        num_workers=os.cpu_count(),
    )

    client.schema.delete_all()

    # Set DPR model used for the Page class
    client.schema.create_class({
        "class": "Page",
        "shardingConfig": {
            "desiredCount": 1,
            "replicas": 3,
        },
        "vectorizer": "text2vec-huggingface",
        "moduleConfig": {
            "passageModel": "sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base",
            "queryModel": "sentence-transformers/facebook-dpr-question_encoder-single-nq-base",
            "options": {
                "waitForModel": True,
                "useGPU": True,
                "useCache": True
            }
        },
        "properties": [
            {
                "name": "url",
                "dataType": ["string"],
            },
            {
                "name": "title",
                "dataType": ["text"],
            },
            {
                "name": "raw",
                "dataType": ["text"],
            },
            {
                "name": "sha",
                "dataType": ["string"],
            }
        ]
    })

    return client


def import_data(client: weaviate.Client):
    start = time.time()
    c=0
    with open(SPHERE_DATASET) as jsonl_file:
        with client.batch as batch:
            for jsonl in jsonl_file:
                json_parsed = json.loads(jsonl)
                batch.add_data_object({
                        'url':  json_parsed['url'],
                        'title': json_parsed['title'],
                        'raw': json_parsed['raw'],
                        'sha': json_parsed['sha']
                    },
                    'Page',
                    json_parsed['id'],
                    vector=json_parsed['vector']
                )
                del json_parsed
                c += 1
                if (c % (BATCH_SIZE * 1000)) == 0:
                    print(f'Imported: {c}, batch_size: {client.batch.recommended_num_objects}')
                    gc.collect()

    end = time.time()
    print('Done in', end - start)

if __name__ == '__main__':
    client = prepare_client()
    import_data(client)
