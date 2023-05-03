from datasets import load_dataset
from utility.DataQA import DataQA
from torch.utils.data import DataLoader
from transformers import default_data_collator

dataset = load_dataset("squad")

#lets sample a small dataset
dataset['train'] = dataset['train'].select([i for i in range(50)])
dataset['validation'] = dataset['validation'].select([i for i in range(5)])

# to make text bold
s_bold = '\033[1m'
e_bold = '\033[0;0m'
train_data = dataset["train"]

for i in range(10):
    # print(train_data[i]['context'])
    print(train_data[i]['answers']['text'])
    # print(train_data[i]['context'].find(train_data[i]['answers']['text'][0]))
    # print(train_data[i]['answers']['answer_start'][0])

for data in train_data:
    print(' ')
    print(s_bold + 'ID -' + e_bold, data['id'])
    print(s_bold +'TITLE - '+ e_bold, data['title'])
    print(s_bold + 'CONTEXT - '+ e_bold,data['context'])
    print(s_bold + 'ANSWERS - ' + e_bold,data['answers']['text'])
    print(s_bold + 'ANSWERS START INDEX - ' + e_bold,data['answers']['answer_start'])
    print(' ')
    break
    
