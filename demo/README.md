# OLIV demo

## Installation
Disclaimer: This demo is only tested on Ubuntu 16.04
pip install -r requirements.txt

## To run
To start demo:
```bash
python demo.py
```
To end demo: press 'q'
To kill process:
```bash
ps aux | grep demo.py | grep -v grep | awk -F ' ' '{print $2}' | sort | head -1 | xargs kill -9
```
