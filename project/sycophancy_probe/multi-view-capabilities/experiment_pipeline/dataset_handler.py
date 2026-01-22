# import kagglehub
# from kagglehub import KaggleDatasetAdapter
from torch.utils.data import DataLoader, Dataset
import random
from typing import List, Dict, Any, Optional
import json
from model_wrapper import ModelWrapper

class DatasetHandler:
    def __init__(self, data_path, model_handler = None, max_len = 128, tokenize = False, data_source = "local", batch_size = 8, split = (0.8, 0.1, 0.1)):
        random.seed(42)
        load_data_funcs = {"local": self.load_local, "kaggle": self.load_kaggle}

        if data_source not in load_data_funcs:
            raise ValueError(f"{data_source} not supported.")
        
        if tokenize and model_handler is None:
            raise ValueError("Model handler must be given to tokenize data")
        
        self.model_handler = model_handler
        self.max_len = max_len
        self.raw_data = load_data_funcs[data_source](data_path, tokenize = tokenize)
        if self.tokenize:
            self.dataset = TextDataset(list(map(self.format_datapt, self.raw_data)))


        else:
            self.dataset = TextDataset(self.raw_data)

        # split into train, val, test & init data loaders
        self.train, self.val, self.test = self.train_val_test_split(split)
        self.train_loader = DataLoader(self.train, batch_size = batch_size)
        self.val_loader = DataLoader(self.val, batch_size = batch_size)
        self.test_loader = DataLoader(self.test, batch_size = batch_size)


    def format_dataset(self, user_key = "question", assist_key = None):
        '''returns a data loader that formats as desired'''
        formatted_data = [self.format_datapt(datapt, user_key, assist_key) for datapt in self.dataset]
        return DataLoader(TextDataset(formatted_data), batch_size = self.batch_size)

         
    def format_datapt(self, datapt, user_key = "question", assist_key = None):
        new_datapt = datapt.copy()
        if assist_key is not None:
            messages = [
                {"role": "user", "content": datapt[user_key]}, 
                {"role": "assistant", "content": datapt[assist_key]}
            ]
        else: # no assistant message
            messages = [
                {"role": "user", "content": datapt[user_key]}
            ]
        formatted_text = self.model_handler.apply_chat_template(messages)
        
        encoding = self.model_handler.tokenizer(
            formatted_text, 
            max_length = self.max_len, 
            padding = 'max_length', 
            truncation = True, 
            return_attention_mask = True, 
            return_tensors = 'pt'
        )

        new_datapt["input_ids"] = encoding["input_ids"].flatten()
        new_datapt["attention_mask"] = encoding["attention_mask"].flatten()

        return new_datapt
    
    def load_local(self, data_path, tokenize = False):
        '''returns a Dataset object'''
        # first turn into a list of dictionaries, then make it a dataset object
        with open(data_path, "r") as f:
            raw_dataset = json.load(f)

        random.shuffle(raw_dataset)

        # if tokenize:
        #     formatted_dataset = list(map(self.format_datapt, raw_dataset))
        #     return TextDataset(formatted_dataset)
        # else:
        #     return TextDataset(raw_dataset)
        return raw_dataset

    
    def load_kaggle(self, data_path, tokenize = False):
        '''returns Dataset object'''
        pass




# def format_data(datapoint, key):
#     '''takes a datapoint with question and answer matching/not matching behavior. Also given a key to specify
#     which behavior in datapoint to use.'''
#     messages = [
#         {"role": "user", "content": datapoint["question"]},
#         {"role": "assistant", "content": datapoint[key]}
#     ]
#     return tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
# class TokenizedTextDataset(Dataset):
#     def __init__(self, data, model_handler):
def train_val_test_split(data, split = (0.8, 0.1, 0.1)):
    n_total = len(data)
    n_train = int(n_total * split[0])
    n_val = int(n_total * split[1])
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]

class TextDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.num_samples = len(data)
        self.data = data

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    model = ModelWrapper("Qwen/Qwen2.5-3B", "cpu")
    print(model)
    dataset = DatasetHandler("test_data.json", model, tokenize=True)
    for datapt in dataset.train: 
        print(datapt)
    
    


