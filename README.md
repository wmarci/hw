# Iris Dataset ML pipeline

Pipeline contains the following stages:
- data preparation (data clean, data split)
- train
- eval
- inference

## Build each stage

data preparation: `docker build -t iris-prep --target prep -f ./prep_train_eval/Dockerfile`

train: `docker build -t iris-train --target train -f ./prep_train_eval/Dockerfile`

eval: `docker build -t iris-eval --target eval -f ./prep_train_eval/Dockerfile`

inference: `docker build -t iris-inference --target inference -f ./inference/Dockerfile`

## Run pipeline

First time: `docker-compose -f iris-pipeline.yaml up --build` (v1.x) `docker compose -f iris-pipeline.yaml up --build` (v2.x)

Not first time/if build is not required: `docker-compose -f iris-pipeline.yaml up` (v1.x) `docker compose -f iris-pipeline.yaml up` (v2.x)

## Inference sample requests

`curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"data": [4.0,3.3,1.7,0.5]}'` {"class_id":0,"class_name":"Iris-setosa"}

`curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"data": [6.2,2.8,4.8,1.8]}'` {"class_id":2,"class_name":"Iris-virginica"}

`curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"data": [7.2,2.8,4.8,1.8]}'` {"class_id":1,"class_name":"Iris-versicolor"}

## Requirements

- docker/docker compose 1.x, 2.x

## Future ideas/tests

- scrips/modules unittests
-
