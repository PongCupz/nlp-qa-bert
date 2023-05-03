import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from utility.dataset import custom_dataset
import warnings
warnings.simplefilter("ignore")

weight_path = "kaporter/bert-base-uncased-finetuned-squad"
# loading tokenizer
tokenizer = BertTokenizer.from_pretrained(weight_path)
#loading the model
model = BertForQuestionAnswering.from_pretrained(weight_path)

n = 1
for item in custom_dataset :
    print("#" * 30)
    print("No.",n)
    context = item['context']
    question = item['question']

    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = tokens.index('[SEP]')

    # we will provide including [SEP] token which seperates question from context and 1 for rest.
    token_type_ids = [0 for i in range(sep_idx+1)] + [1 for i in range(sep_idx+1,len(tokens))]

    # Run our example through the model.
    out = model(torch.tensor([input_ids]), # The tokens representing our input text.
                    token_type_ids=torch.tensor([token_type_ids]))

    start_logits,end_logits = out['start_logits'],out['end_logits']
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_logits)
    answer_end = torch.argmax(end_logits)

    answer = ''.join(tokens[answer_start:answer_end])

    print(f"Question: {question}")
    print(f"Correnct Answer: {item['answer']}")
    print(f"Predict Answer: {answer}")
    n = n +1