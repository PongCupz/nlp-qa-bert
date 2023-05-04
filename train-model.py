import torch,random,time
import numpy as np
import warnings
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, DistilBertForQuestionAnswering, default_data_collator, AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AdamW

from utility.train import  train
from utility.DataQA import DataQA
from utility.utils import format_time, predict_answers_and_evaluate, preprocess_validation_examples, qa_loss_fn
from config import config, loss_config
from datasets import Dataset
# from utility.dataset import custom_datasets
warnings.simplefilter("ignore")



h = {
    "train" : {
        "loss" : [],
    },
    "validation" : {
        "loss" : [],
    }
}
epochs = config["epochs"]

checkpoint = config["checkpoint"]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#lets sample a small dataset
dataset = load_dataset("squad")
dataset['train'] = dataset['train'].select([i for i in range(8000)])
dataset['validation'] = dataset['validation'].select([i for i in range(2000)])

# dataset = custom_datasets


train_dataset = DataQA(dataset,mode="train")
val_dataset = DataQA(dataset,mode="validation")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=config["batch_size"],
)
eval_dataloader = DataLoader(
    val_dataset, collate_fn=default_data_collator, batch_size=config["batch_size"]
)

# model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs


validation_processed_dataset = dataset["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns = dataset["validation"].column_names,
)

# to reproduce results
seed_val = config["seed_val"]
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#storing all training and validation stats
stats = []


#to measure total training time
total_train_time_start = time.time()

for epoch in range(epochs):
    print(' ')
    print(f'=====Epoch {epoch + 1}=====')
    print('Training....')
    
    # ===============================
    #    Train
    # ===============================   
    # measure how long training epoch takes
    t0 = time.time()
    
    training_loss = 0
    # loop through train data
    model.train()
    for step,batch in enumerate(train_dataloader):
        
        # we will print train time in every 40 epochs
        if step%40 == 0 and not step == 0:
            elapsed_time = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed_time))

        
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
            


        #set gradients to zero
        model.zero_grad()

        result = model(input_ids = input_ids, 
                        attention_mask = attention_mask,
                        start_positions = start_positions,
                        end_positions = end_positions,
                        return_dict=True)
        
        loss = result.loss
    
        #accumulate the loss over batches so that we can calculate avg loss at the end
        training_loss += loss.item()      

        #perform backward prorpogation
        loss.backward()

        # update the gradients
        optimizer.step()

    # calculate avg loss
    avg_train_loss = training_loss/len(train_dataloader) 

    # calculates training time
    training_time = format_time(time.time() - t0)
    
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    h['train']["loss"].append(avg_train_loss)
    
    # ===============================
    #    Validation
    # ===============================
    
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    
    ignored_index =tokenizer.model_max_length

    validation_loss = 0
    start_logits,end_logits = [],[]
    for step,batch in enumerate(eval_dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():  
            result = model(input_ids = input_ids, 
                        attention_mask = attention_mask,return_dict=True)

        start_logits.append(result.start_logits.cpu().numpy())
        end_logits.append(result.end_logits.cpu().numpy())
        
        ########## Validation Loss ##########
        loss = qa_loss_fn(result.start_logits,result.end_logits, start_positions, end_positions, loss_config)
        validation_loss += loss.item()
        ########## Validation Loss ##########

    # calculate avg loss
    avg_validation_loss = validation_loss/len(eval_dataloader) 

    print("")
    print("  Average validation loss: {0:.2f}".format(avg_validation_loss))
    h['validation']["loss"].append(avg_validation_loss)

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)


    # calculating metrics
    # answers,metrics_ = predict_answers_and_evaluate(start_logits,end_logits,validation_processed_dataset,dataset["validation"])
    # print(f'Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')

    # print('answers')
    # print(answers)
    print('')
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation took: {:}".format(validation_time))

    if epoch > 1:
        print(abs(h['train']["loss"][epoch] - h['train']["loss"][epoch -1]) / h['train']["loss"][epoch -1])
        if abs(h['train']["loss"][epoch] - h['train']["loss"][epoch -1]) / h['train']["loss"][epoch -1] < 0.001 :
            break

tokenizer.save_pretrained("models/tokenizer/")
model.save_pretrained("models/tokenizer/")

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_train_time_start)))

print(h["train"]["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(1, epochs + 1), h["train"]["loss"], label="train_loss")
plt.plot(np.arange(1, epochs + 1), h["validation"]["loss"], label="val_loss")
plt.title(f"Training & Validation Loss")

plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"models/loss.png")
