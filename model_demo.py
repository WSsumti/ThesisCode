import torch as torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

from transformers import AutoModel

import os

cwd = os.getcwd()

# PhoBERT vanilla model
class PhoBERT_Vanilla(nn.Module):
    def __init__(self, n_classes):
        super(PhoBERT_Vanilla, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        self.drop = nn.Dropout(p=0.3)

        #self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(768, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        #lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(x)
        return x
    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob

# PhoBERT LSTM model
class PhoBERT_LSTM(nn.Module):
    def __init__(self, n_classes):
        super(PhoBERT_LSTM, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        self.drop = nn.Dropout(p=0.3)

        self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(128, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(lstm_output)
        return x
    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob

# VisoBERT Model
class VisoBERT(nn.Module):
    def __init__(self, n_classes):
        super(VisoBERT, self).__init__()
        self.bert = AutoModel.from_pretrained('uitnlp/visobert')
        self.drop = nn.Dropout(p=0.3)

        #self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(768, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        #lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(x)
        return x
    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob
    
# VisoBERT ALRL
class VisoBERT_ALRL(nn.Module):
    def __init__(self, n_classes, freeze_bert=True, freeze_lstm=True):
        super(VisoBERT_ALRL, self).__init__()
        self.bert = AutoModel.from_pretrained('uitnlp/visobert')
        self.drop = nn.Dropout(p=0.3)

        #         self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(768, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        #         lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(x)
        return x

    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob

# PhoBERT LSTM ALRL
class PhoBERT_LSTM_ALRL(nn.Module):
    def __init__(self, n_classes, freeze_bert=True, freeze_lstm=True):
        super(PhoBERT_LSTM_ALRL, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        self.drop = nn.Dropout(p=0.3)

        self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(128, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        if freeze_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(lstm_output)
        
        return x

    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob

#PhoBERT Vanilla ALRL
class PhoBERT_ALRL(nn.Module):
    def __init__(self, n_classes, freeze_bert=True, freeze_lstm=True):
        super(PhoBERT_ALRL, self).__init__()
        self.bert = AutoModel.from_pretrained('vinai/phobert-base')
        self.drop = nn.Dropout(p=0.3)

        #         self.lstm = nn.LSTM(num_layers=1, batch_first=True, input_size= 768 ,hidden_size= 128)

        self.fc = nn.Linear(768, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False  # Dropout will errors if without this
        )

        x = self.drop(output)

        #         lstm_output, (hidden_state, cell_state) = self.lstm(x)

        x = self.fc(x)
        return x

    def act(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        #         logits = logits.cpu().detach()
        pd = Categorical(logits=logits)
        action = pd.sample()
        log_prob = pd.log_prob(action)
        return logits, action, log_prob
    
# Datasets
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=120):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        To customize dataset, inherit from Dataset class and implement
        __len__ & __getitem__
        __getitem__ should return
            data:
                input_ids
                attention_masks
                text
                targets
        """
        row = self.df.iloc[index]
        text, label = self.get_input_data(row)

        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }

    def labelencoder(self, text):
        if text == 'Enjoyment':
            return 0
        elif text == 'Disgust':
            return 1
        elif text == 'Sadness':
            return 2
        elif text == 'Anger':
            return 3
        elif text == 'Surprise':
            return 4
        elif text == 'Fear':
            return 5
        else:
            return 6

    def get_input_data(self, row):
        # Preprocessing: {remove icon, special character, lower}
        text = row['Sentence']
        #text = ' '.join(simple_preprocess(text))
        label = self.labelencoder(row['Emotion'])

        return text, label
