import torch
from torch.utils.data import Dataset
from utility.utils import train_data_preprocess, preprocess_validation_examples

class DataQA(Dataset):
    def __init__(self, dataset,mode="train"):
        self.mode = mode
        
        if self.mode == "train":
            # sampling
            self.dataset = dataset["train"]
            self.data = self.dataset.map(
                train_data_preprocess,
                batched=True,remove_columns= dataset["train"].column_names
            )
        
        else:
            self.dataset = dataset["validation"]
            self.data = self.dataset.map(
                train_data_preprocess,
                batched=True,remove_columns = dataset["validation"].column_names,
            )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        out = {}
        example = self.data[idx]
        out['input_ids'] = torch.tensor(example['input_ids'])
        out['attention_mask'] = torch.tensor(example['attention_mask'])

        out['start_positions'] = torch.unsqueeze(torch.tensor(example['start_positions']),dim=0)
        out['end_positions'] = torch.unsqueeze(torch.tensor(example['end_positions']),dim=0)
        
        return out