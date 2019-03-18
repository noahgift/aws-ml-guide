setup:
	python3 -m venv ~/.aws-ml-guide
	source ~/.aws-ml-guide/bin/activate

makesite:
	jupyter-book create ../TEMPLATE/aws-ml-guide

buildsite:
	jupyter-book build ../TEMPLATE/aws-ml-guide
	cp -R ../TEMPLATE/aws-ml-guide/* ../
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
