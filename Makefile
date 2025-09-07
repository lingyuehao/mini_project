install:
	 pip install --upgrade pip &&\
	 pip install -r requirements.txt

format:
	 black *.py

lint:
	 flake8 *.py

test:
	 pytest -vv --cov=. --maxfail=1

clean:
	 rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test

