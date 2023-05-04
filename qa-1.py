import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, RobertaTokenizer, DistilBertTokenizer
import warnings
from utility.dataset import custom_dataset
warnings.simplefilter("ignore")

# weight_path = "kaporter/bert-base-uncased-finetuned-squad"
# weight_path = "bert-large-uncased-whole-word-masking-finetuned-squad"

# tokenizer = AutoTokenizer.from_pretrained(weight_path)
# model = AutoModelForQuestionAnswering.from_pretrained(weight_path)

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer/")
model = AutoModelForQuestionAnswering.from_pretrained("models/tokenizer/")

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
    print(f"Correnct Answer: {item['answer']}")
    print(f"Predict Answer: {answer}")
    n = n +1