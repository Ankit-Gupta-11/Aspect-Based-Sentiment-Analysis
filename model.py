import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import warnings
import pickle
from CustomDataset import dataset

# with "dataset-new" as f:
#     dataset = pickle.load(f)


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
    MAX_LEN = 20
    MODEL_PATH = "./Models/MyModel.pt"

class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        # self.embedding = embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True)
        self.fc1 = nn.Linear(2*hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()


        
    def forward(self, text):
        
        max_len, N = text.shape
        hidden = torch.zeros((2, N , self.hidden_dim),
                          dtype=torch.float)
        cell = torch.zeros((2, N , self.hidden_dim),
                            dtype=torch.float)

        hidden = hidden.to(config.DEVICE)
        embedded = self.embedding(text)
        output, hidden = self.lstm(embedded, (hidden, cell))
        y_pred = output[-1,:,:]
        y_pred = self.fc1(y_pred)
        y_pred = self.fc2(y_pred)
        y_pred = self.sigmoid(y_pred)
                         
        return y_pred  


VOCAB_SIZE = len(dataset.source_vocab)
HIDDEN_DIM = 128
OUTPUT_DIM = 1
VOCAB = list(dataset.source_vocab.stoi)
# VOCAB_SIZE = 2175


# embedding_layer = get_emb_layer_with_weights(target_vocab = VOCAB, emb_model = fasttext_model, trainable = False)

model = Model(VOCAB_SIZE, config.EMB_DIM, HIDDEN_DIM, OUTPUT_DIM)
model = model.to(config.DEVICE)

pretrained = torch.load("Model/My-Model.pt", map_location = config.DEVICE)

def get_prediction(text):
    model.load_state_dict(pretrained["LSTM-Model-for-0"])
    # model1 = model.load_state_dict(pretrained["RNN-Model-for-1"])
    # model2 = model.load_state_dict(pretrained["RNN-Model-for-2"])
    # model3 = model.load_state_dict(pretrained["RNN-Model-for-3"])
    # model4 = model.load_state_dict(pretrained["RNN-Model-for-4"])
    model.eval()
    result = model(text)
    return result
    