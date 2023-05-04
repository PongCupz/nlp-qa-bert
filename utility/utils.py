import numpy as np
import datetime
import collections
import evaluate
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from config import config

trained_checkpoint = config["checkpoint"]
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

def train_data_preprocess(examples):
    
    """
    generate start and end indexes of answer in context
    """

    def find_context_start_end_index(sequence_ids):
        """
        returns the token index in whih context starts and ends
        """
        token_idx = 0
        while sequence_ids[token_idx] != 1:  #means its special tokens or tokens of queston
            token_idx += 1                   # loop only break when context starts in tokens
        context_start_idx = token_idx
    
        while sequence_ids[token_idx] == 1:
            token_idx += 1
        context_end_idx = token_idx - 1
        return context_start_idx,context_end_idx  
    
    
    questions = [q.strip() for q in examples["question"]]
    context = examples["context"]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        context,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,  #returns id of base context
        return_offsets_mapping=True,  # returns (start_index,end_index) of each token
        padding="max_length"
    )


    start_positions = []
    end_positions = []

    
    for i,mapping_idx_pairs in enumerate(inputs['offset_mapping']):
        context_idx = inputs['overflow_to_sample_mapping'][i]
    
        # from main context
        answer = answers[context_idx]
        answer_start_char_idx = answer['answer_start'][0]
        answer_end_char_idx = answer_start_char_idx + len(answer['text'][0])

    
        # now we have to find it in sub contexts
        tokens = inputs['input_ids'][i]
        sequence_ids = inputs.sequence_ids(i)
   
        # finding the context start and end indexes wrt sub context tokens
        context_start_idx,context_end_idx = find_context_start_end_index(sequence_ids)
    
        #if the answer is not fully inside context label it as (0,0)
        # starting and end index of charecter of full context text
        context_start_char_index = mapping_idx_pairs[context_start_idx][0]
        context_end_char_index = mapping_idx_pairs[context_end_idx][1]
    

        #If the answer is not fully inside the context, label is (0, 0)
        if (context_start_char_index > answer_start_char_idx) or (
            context_end_char_index < answer_end_char_idx):
            start_positions.append(0)
            end_positions.append(0)
    
        else:

            # else its start and end token positions
            # here idx indicates index of token
            idx = context_start_idx
            while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                idx += 1
            start_positions.append(idx - 1)  
        

            idx = context_end_idx
            while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
    


def preprocess_validation_examples(examples):
    """
    preprocessing validation data
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")

    base_ids = []

    for i in range(len(inputs["input_ids"])):
        
        # take the base id (ie in cases of overflow happens we get base id)
        base_context_idx = sample_map[i]
        base_ids.append(examples["id"][base_context_idx])
        
        # sequence id indicates the input. 0 for first input and 1 for second input
        # and None for special tokens by default
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        # for Question tokens provide offset_mapping as None
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["base_id"] = base_ids
    return inputs



def predict_answers_and_evaluate(start_logits,end_logits,eval_set,examples):
    """
    make predictions 
    Args:
    start_logits : strat_position prediction logits
    end_logits: end_position prediction logits
    eval_set: processed val data
    examples: unprocessed val data with context text
    """
    # appending all id's corresponding to the base context id
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["base_id"]].append(idx)

    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # looping through each sub contexts corresponding to a context and finding
        # answers
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]
        
            # sorting the predictions of all hidden states and taking best n_best prediction
            # means taking the index of top 20 tokens
            start_indexes = np.argsort(start_logit).tolist()[::-1][:n_best]
            end_indexes = np.argsort(end_logit).tolist()[::-1][:n_best]
        
    
            for start_index in start_indexes:
                for end_index in end_indexes:
                
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                       ):
                        continue

                    answers.append({
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        })

    
            # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    
    metric = evaluate.load("squad")

    theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    
    metric_ = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return predicted_answers,metric_

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def ce_loss(
    pred, truth, smoothing=False, trg_pad_idx=-1, eps=0.1
):
    """
    Computes the cross entropy loss with label smoothing

    Args:
        pred (torch tensor): Prediction
        truth (torch tensor): Target
        smoothing (bool, optional): Whether to use smoothing. Defaults to False.
        trg_pad_idx (int, optional): Indices to ignore in the loss. Defaults to -1.
        eps (float, optional): Smoothing coefficient. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    truth = truth.contiguous().view(-1)

    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)

    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    loss = -one_hot * F.log_softmax(pred, dim=1)

    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)

    return loss.sum()

def qa_loss_fn(start_logits, end_logits, start_positions, end_positions, config):
    """
    Loss function for the question answering task.
    It is the sum of the cross entropy for the start and end logits

    Args:
        start_logits (torch tensor): Start logits
        end_logits (torch tensor): End logits
        start_positions (torch tensor): Start ground truth
        end_positions (torch tensor): End ground truth
        config (dict): Dictionary of parameters for the CE Loss.

    Returns:
        torch tensor: Loss value
    """
    bs = start_logits.size(0)

    start_loss = ce_loss(
        start_logits,
        start_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
    )

    end_loss = ce_loss(
        end_logits,
        end_positions,
        smoothing=config["smoothing"],
        eps=config["eps"],
    )

    total_loss = start_loss + end_loss

    return total_loss / bs
