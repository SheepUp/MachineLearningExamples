.PHONY: clean

all:
	rm -rfv ./env
	virtualenv ./env -p python2 --no-site-packages
	./env/bin/pip install --upgrade pip
	./env/bin/pip install -r requirements.txt

clean:
	rm -rfv bin develop-eggs dist downloads eggs env parts
	rm -fv .DS_Store .coverage .installed.cfg bootstrap.py
	find . -name '*.pyc' -exec rm -fv {} \;
	find . -name '*.pyo' -exec rm -fv {} \;
	find . -depth -name '*.egg-info' -exec rm -rfv {} \;
	find . -depth -name '__pycache__' -exec rm -rfv {} \;
