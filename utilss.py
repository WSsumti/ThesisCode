import re
from bs4 import BeautifulSoup
import pandas as pd
from model_demo import SentimentDataset
from torch.utils.data import Dataset, DataLoader

class Cleaner():
    def __init__(self):
        pass

    def put_line_breaks(self, text):
        text = text.replace('</p>', '</p>\n')
        return text

    def remove_html_tags(self, text):
        cleantext = BeautifulSoup(text, "lxml").text
        return cleantext

    def remove_redundant_space(self, text):
        return re.sub(r'\s+', ' ', text)

    def remove_redundant_end_letters(self, text):
        pattern = re.compile(r'(\w+?)(\w)\2+$')
        return pattern.sub(r'\1\2', text)

    def remove_redundant_sentece(self, text):
        removed = ''
        for letter in text.split():
            removed += self.remove_redundant_end_letters(letter) + ' '
        return removed

    def clean(self, text):
        text = text.lower()
        text = self.remove_redundant_space(text)
        text = self.put_line_breaks(text)
        #text = self.remove_html_tags(text)
        text = self.remove_redundant_sentece(text)
        return text
    
class Utilss():
    def __init__(self):
        pass
    def get_data(path):
        df = pd.read_excel(path, sheet_name=None)['Sheet1']
        df.columns = ['index', 'Emotion', 'Sentence']
        # unused column
        df.drop(columns=['index'], inplace=True)
        return df
    def prepare_loaders(df, fold, tokenizer):
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        train_dataset = SentimentDataset(df_train, tokenizer, max_len=120)
        valid_dataset = SentimentDataset(df_valid, tokenizer, max_len=120)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=2)

        return train_loader, valid_loader
    