from data_utils import WordData
from encoder_decoder import Encoder, Decoder, EncoderDecoder
from sigmorphon_data_utils import load_original_data, flatten_data
from training_pipeline import train
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch

TASK_FOLDER = "/Users/oleg/2019/task1/"

tur_tat_data = flatten_data(load_original_data(('turkish', 'tatar'), TASK_FOLDER))
tur_tat_df = pd.DataFrame(tur_tat_data)
tur_tat_df.head()

tur_df = tur_tat_df.loc[tur_tat_df['lang'] == 'turkish']
tur_df.head()


def initialize_smth2index():
    smth2index = defaultdict(lambda: len(smth2index))
    return smth2index


tag2index = initialize_smth2index()

# noinspection PyStatementEffect
[tag2index[tag] for tags in list(tur_df.tags) for tag in tags]

char2index = initialize_smth2index()

# noinspection PyStatementEffect
[char2index[char] for lem in list(tur_df.lem) for char in lem]

wf_pad_token = 'CPAD'
wf_pad_token_index = char2index[wf_pad_token]

train_df, test_df = train_test_split(tur_df, test_size=0.2)
lem_test, tags_test, wf_test = test_df.lem, test_df.tags, test_df.wf

lem_train, lem_val, tags_train, tags_val, wf_train, wf_val = train_test_split(
    train_df.lem, train_df.tags, train_df.wf, test_size=0.1)

train_dataset = WordData(list(tags_train), list(lem_train), list(wf_train), tag2index, char2index)
train_loader = DataLoader(train_dataset, batch_size=64)

validation_dataset = WordData(list(tags_val), list(lem_val), list(wf_val), tag2index, char2index)
validation_loader = DataLoader(validation_dataset, batch_size=64)

test_dataset = WordData(list(tags_test), list(lem_test), list(wf_test), tag2index, char2index)
test_loader = DataLoader(test_dataset, batch_size=64)


encoder = Encoder(len(tag2index), len(char2index))
decoder = Decoder(encoder.hidden_size, len(char2index))
model = EncoderDecoder(wf_pad_token_index, encoder, decoder)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = model.to(device)
encoder = encoder.to(device)
decoder = decoder.to(device)


N_EPOCHS = 10

train(model, N_EPOCHS, train_loader, test_loader, validation_loader, device)