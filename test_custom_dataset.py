from datasets import Dataset
from utility.dataset import contexts,questions,answers

dataset = {}
dataset['train'] = Dataset.from_dict({'question':questions[0:25], 'context':contexts[0:25], 'answers':answers[0:25]})
dataset['validation'] = Dataset.from_dict({'question':questions[25:30], 'context':contexts[25:30], 'answers':answers[25:30]})
# print(dataset)


for i in range(len(contexts)):
    print(contexts[i].find(answers[i]))

