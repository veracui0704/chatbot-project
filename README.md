# Project Title

ChatBot trained with movie subtitles for self-learning and practices.  

The the model flow mainly follows the user **Currie32** with several modifications(__combination with API and html__).  
Please refer to [his Github work](https://github.com/Currie32/Chatbot-from-Movie-Dialogue/blob/master/Chatbot_Attention.ipynb) with for your detailed information.  

RNN and LSTM are used in the program to perform seq2seq modelling with attention(many inputs-many outputs) and avoid long-term memory problem.

## Getting Started

The following simple instructions will give you some basic introduction to this project which you could download and try on your computer for studying and testing purposes.  
You may have a look at the list of documents below and their purposes.

- data.zip: data used, please download and unzip it into a folder "data"  
- chatbot.py: main program  
- app.py + template/chatbothtml.html: try to use html and flask to call python by clicking on a button  
- server.py: try to call program by API

### Dependencies
1. For chatbot.py:
```
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
```
2. For the interaction with API and html:
```
from flask import Flask, request
from flask_restful import Resource, Api
```
```
from flask import Flask, request, render_template
```
Please note that to run chatbot.py, it is necessary that you check the version of your current Python and TensorFlow module.  
Here we have
```
Python 3.5, Tensorflow 1.0.0
```
to support specific functions used in the model, using virtual environment is also encouraged when testing the program

## Running the program

To train the model:
```
python chatbot.py train
```
To get predictions or chat with the bot:
```
python chatbot.py predict
```
