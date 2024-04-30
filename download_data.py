import requests
from zipfile import ZipFile
import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-p", "--path", dest="path", help="Path where save folder", default="/")
command_line_args = parser.parse_args()
data_path = command_line_args.path

URL = 'http://dl.dropboxusercontent.com/scl/fi/2j7nx8y3jbyypdm7r100f/dataset.zip?rlkey=veu69cngj0els6emgp549r06u&dl=0'

r = requests.get(URL)
os.makedirs(os.path.join(data_path, 'tmp'), exist_ok=True)
with open(os.path.join(data_path, 'tmp/dataset.zip'), 'wb') as f:
    f.write(r.content)
  
os.makedirs(os.path.join(data_path, 'data'), exist_ok=True)
with ZipFile(os.path.join(data_path, 'tmp/dataset.zip'), 'r') as zObject:
    zObject.extractall(path=os.path.join(data_path, 'data'))

shutil.rmtree(os.path.join(data_path, 'tmp'))
