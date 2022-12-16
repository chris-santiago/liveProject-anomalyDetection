import csv
import requests
import random

FILE = './jupyter/data/test.csv'


def read_data(file):
	with open(file) as csvfile:
		reader = csv.reader(csvfile)
		next(reader)  # skip header
		for row in reader:
			yield [float(x) for x in row]


def request_prediction(data, score=True):
	url = f'http://localhost:8000/prediction?score={str(score).lower()}'
	return requests.post(url=url, json=data)


def request_info():
	url = 'http://localhost:8000/model_information'
	return requests.get(url=url)


def request_all(data, score=True):
	resp = request_prediction(data=data, score=score)
	request_info()
	return resp.json()


if __name__ == '__main__':
	data = read_data(FILE)
	responses = [request_all(d, bool(random.getrandbits(1))) for d in data]
	for r in responses[:20]:
		print(r)
