setup:
	#You may want to create an alias to automatically source this:
	python3 -m venv ~/.aws-ml-guide

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint --disable=R,C *.py


