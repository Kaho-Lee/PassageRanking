from utils import *
from DataGenerator import dataPipeLine
import requests
from requests import get
import os
import shutil


#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
    
def download_file_from_google_drive(id, destination, root_path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination, root_path)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, root_path):
    CHUNK_SIZE = 32768

    with open(root_path+destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#Taken from https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
def download_fromURL(url, file_name, root_path):
    # open in binary mode
    with open(root_path+file_name, "wb") as file:
        # get request
        response = get(url)
        print('file get, writing to path')
        # write to file
        file.write(response.content)


if __name__=="__main__":

    download = False # download all the dataset needed for implementation
    init = False #Extracting the dowloaded file, and convert to the format needed by implementation
    root_path = '../Data/'

    if download:
        
        if(os.path.exists(root_path)):
            shutil.rmtree(root_path)
            print('Creating data storage folder')
        os.makedirs(root_path)
        
        print('Downloading dataset and pre-trained embedding')
        file_id = '1npkPA-BdiGELHfBrUOcpqumjbQTspg9p'
        destination = 'part2.zip'
        download_file_from_google_drive(file_id, destination, root_path)

        glove_url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        destination = 'glove.6B.zip'
        download_fromURL(glove_url, destination, root_path)
        print('Please mannually unzip the file listed in the following three line and\
            place them into root path')

    
    pre_trained_model_path = root_path+'glove.6B.50d.txt'
    train_path = root_path+'train_data.tsv'
    val_path = root_path+'validation_data.tsv'

    gloveEmbedding_path = root_path+'gloveEmbedding.json'
    idf_val = root_path+'val_idf.json'
    idf_train = root_path+'train_idf.json'

    if init:
        print('Extract the neccessary data for code implementation')
        ExtractPreTrained(pre_trained_model_path, gloveEmbedding_path)
        IDFOfDataSet(train_path, idf_train)
        IDFOfDataSet(val_path, idf_val)
