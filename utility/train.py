import torch
import random,time
import numpy as np
from utility.utils import format_time, predict_answers_and_evaluate, preprocess_validation_examples

def train(model, train_dataloader, eval_dataloader, dataset, device, optimizer, epochs):
    # we need processed validation data to get offsets at the time of evaluation
    validation_processed_dataset = dataset["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns = dataset["validation"].column_names,
    )

    # to reproduce results
    seed_val = 42
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
        
        
        # ===============================
        #    Validation
        # ===============================
        
        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        

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
        # start_logits = start_logits[: len(val_dataset)]
        # end_logits = end_logits[: len(val_dataset)]




        # calculating metrics
        answers,metrics_ = predict_answers_and_evaluate(start_logits,end_logits,validation_processed_dataset,dataset["validation"])
        print(f'Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')

        print('answers')
        print(answers)
        print('')
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation took: {:}".format(validation_time))

    torch.save(model.state_dict(), 'models/bert.pth')

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_train_time_start)))