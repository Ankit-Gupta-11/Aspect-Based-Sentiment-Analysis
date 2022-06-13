import torch
import numpy as np
import torch
import warnings
import warnings
import pickle
from preprocessing import preprocessing
from gensim.models import FastText 
import model as my_model
from CustomDataset import dataset, fasttext_model

# fasttext_model = FastText.load("FastText-Embeddings/FastText-Model-For-ABSA.bin")

from fastapi import FastAPI 

app = FastAPI()

class config:
    warnings.filterwarnings("ignore", category = UserWarning)
    IMG_SIZE = (224,224)
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
    FOLDS = 5
    SHUFFLE = True
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 30
    EMB_DIM = 100
    MAX_LEN = 10
    MODEL_PATH = "./Models/MyModel.pt"


# Loading dataset object used to train the network using Pickle module 

# with "dataset" as f:
#     dataset = pickle.load(f)



def numericalize(text):
    
    numerialized_source = [] 
    numerialized_source = [dataset.source_vocab.stoi["<SOS>"]]
    numerialized_source += dataset.source_vocab.numericalize(text)
    numerialized_source.append(dataset.source_vocab.stoi["<EOS>"])
    
    return numerialized_source

def padding(source):
    padded_sequence = torch.zeros(config.MAX_LEN, 1, dtype = torch.int)
    source = torch.tensor(source)
    
    if len(source) > config.MAX_LEN:
        padded_sequence[:, 0] = source[: config.MAX_LEN]
    else:
        padded_sequence[:len(source), 0] = padded_sequence[:len(source), 0] + source
    
    return padded_sequence

def infer_processing(text):
    
    text = preprocessing(text)
    text = numericalize(text)
    text = padding(text)
    return text

aspects = ["phone", "camera", "battery", "neutral", "processor"]

def get_similarity(text, aspect):
    try:
#         text = " ".join(text)
        return fasttext_model.wv.n_similarity(text, aspect)
    except:
        return 0
    
def best_aspect(text, aspects):
    a = []
    
    for aspect in aspects:
        a.append(get_similarity(text, aspect))
    
    return aspects[np.argmax(a)]


sample = "good best great good best great good best great good best great good best great"

ba = best_aspect(preprocessing(sample), aspects)

a = infer_processing(sample).to(config.DEVICE)

results = my_model.get_prediction(a)


@app.get('/')
def welcome_msg():
    return "Hello welcome to our API on Aspect Based Sentiment Analysis"

@app.post("/predict")
def predict(comment : str):
    aspect = best_aspect(preprocessing(comment), aspects)
    text = infer_processing(comment).to(config.DEVICE)
    sentiment = my_model.get_prediction(text)
    sentiment = sentiment.cpu().detach().numpy()[0]

    if sentiment > 0.5:
        sentiment = 'Positively'
    else :
        sentiment = 'Negatively'
    return {"Best Aspect" : aspect, "Sentiment" : sentiment, "Review" : comment, "Observation" : f"The reviewer is talking {sentiment} about the {aspect} of the phone in his/her comment"}



