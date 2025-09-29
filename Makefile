install:
	 pip install --upgrade pip &&\
	 pip install -r requirements.txt

format:
	 black .

lint:
	 flake8 --ignore=E501 .

test:
	 pytest -vv --cov=. --maxfail=1

clean:
	 rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test

