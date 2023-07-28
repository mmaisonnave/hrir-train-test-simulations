from lxml import etree
from bs4 import BeautifulSoup
import os

import sys
sys.path.append('')

def get_text(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    if root.find('.//HiddenText') is not None:
        text = (root.find('.//HiddenText').text)

    elif root.find('.//Text') is not None:
        text = (root.find('.//Text').text)

    else:
        text = ''
    return BeautifulSoup(text, parser='html.parser', features="lxml").get_text().replace('>','').replace('<','')


def get_title(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
   
    title = root.find('.//Title').text
    str_=''
    if not title is None:
        str_+=f'{title}.'

    return BeautifulSoup(str_, parser='html.parser', features="lxml").get_text().replace('>','').replace('<','')

def get_date(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    date = root.find('.//NumericDate').text
    return date


def get_title_and_text(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    if root.find('.//HiddenText') is not None:
        text = (root.find('.//HiddenText').text)

    elif root.find('.//Text') is not None:
        text = (root.find('.//Text').text)

    else:
        text = None
    title = root.find('.//Title').text
    str_=''
    if not title is None:
        str_+=f'{title}.'
    if not text is None:
        str_+=f'{text}.'

    return BeautifulSoup(str_, parser='html.parser', features="lxml").get_text()



def get_filename(id_):
    root ='/home/ec2-user/SageMaker/data'
    data_sources = [os.path.join(root, folder) for folder in os.listdir(root)]
    for data_source in data_sources:
        if os.path.isfile(os.path.join(data_source,id_+'.xml')):
            return os.path.join(data_source,id_+'.xml')
        
