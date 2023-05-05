import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
import warnings
from utility.dataset import custom_dataset, custom_datasets
from utility.DataQA import DataQA
from torch.utils.data import DataLoader

from config import config
from utility.utils import predict_answers_and_evaluate, preprocess_validation_examples

warnings.simplefilter("ignore")

weight_path = "models/tokenizer/"

tokenizer = AutoTokenizer.from_pretrained(weight_path)
model = AutoModelForQuestionAnswering.from_pretrained(weight_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n = 1
for item in custom_dataset :
    print("#" * 30)
    print("No.",n)
    context = item['context']
    question = item['question']
    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
    start_scores, end_scores = model(**inputs, return_dict=False)
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)
    answer_tokens = inputs["input_ids"][0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens)
    print(f"Question: {question}")
    print(f"Target Answer: {item['answer']}")
    print(f"Predict Answer: {answer}")
    n = n +1

data_set = {"validation":  custom_datasets["all"]}
validation_processed_dataset = data_set["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns = data_set["validation"].column_names,
)
val_dataset = DataQA(data_set,mode="validation")
eval_dataloader = DataLoader(
    val_dataset, collate_fn=default_data_collator, batch_size=config["batch_size"]
)

start_logits,end_logits = [],[]
for step,batch in enumerate(eval_dataloader):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():  
        result = model(input_ids = input_ids, 
                    attention_mask = attention_mask,return_dict=True)

    start_logits.append(result.start_logits.cpu().numpy())
    end_logits.append(result.end_logits.cpu().numpy())

start_logits = np.concatenate(start_logits)
end_logits = np.concatenate(end_logits)

answers,metrics_ = predict_answers_and_evaluate(start_logits,end_logits,validation_processed_dataset,data_set["validation"])
print('Exact match: {:.2f}'.format(metrics_["exact_match"]))
print('F1 score: {:.2f}'.format(metrics_["f1"]))