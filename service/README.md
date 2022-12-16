# Making Predictions

## Request prediction with anomaly score

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/prediction?score=true' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  -0.15795269682074692,-0.10674898283572026
]'
```

## Request prediction w/o anomaly score

```bash
curl -X 'POST' \
  'http://0.0.0.0:8000/prediction?score=false' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '[
  -0.15795269682074692,-0.10674898283572026
]'
```

## Request model info

```bash
curl -X 'GET' \
  'http://0.0.0.0:8000/model_information' \
  -H 'accept: application/json'
```