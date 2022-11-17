from requests import session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
import json
from tqdm import tqdm
import pickle
import logging
from tree_sitter import Language, Parser
logging.basicConfig(level=logging.INFO)
import  argparse
import math
from get_code_label import get_prompt_label, parse_code
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset, Features 
from transformers import AutoModelForSequenceClassification
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import  argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to extended logs frame', required=True) # change to True
parser.add_argument('-c', '--cudadevice', help='cuda device id', default=0, required=True, type=int)
parser.add_argument('-b', '--batchsize', help='batch size', default=1000, required=True, type=int)
parser.add_argument('-o', '--output', help='Output path of .pkl file', required=True) # change to True
parser.add_argument('-e', '--embedding', help='Whether to get embeddings for suggestion and prompt', required=True, type=int)
parser.add_argument('-m', '--maxusers', help='max users', default=100, required=True, type=int)
parser.add_argument('-a', '--onlyacceptreject', help='only get features for accept reject events (1 if yes 0 ow)', default=0, required=False, type=int)



def get_embedding_list(list_of_strs, batch_size=16):
    def tokenize_function_embedding(examples):
        prompt_token = tokenizer(examples['text'], return_tensors="pt",  padding="max_length", truncation=True )['input_ids']
        encoded_tokens = model(prompt_token.to(device)).pooler_output.detach().cpu().numpy()
        dict = {'encoded_tokens': encoded_tokens}
        return dict# overall_tokens


    #a = df_observations[0][0].CurrentPrompt.to_numpy()

    dataset = Dataset.from_dict({"text": list_of_strs })

    ds_train_tokenized = dataset.map(tokenize_function_embedding, batched= True, batch_size=batch_size)
    embeddings = [ds_train_tokenized[i]['encoded_tokens'] for i in range(len(ds_train_tokenized))]
    return embeddings

def text_features(list_of_strs):
    list_of_features = []
    for str in list_of_strs:
        numb_of_words = len(str.split())
        # if includes #
        includes_hash = '#' in str
        # includes 'print'
        includes_print = 'print' in str
        # includes '='
        includes_equal = '=' in str or '<=' in str or '>=' in str or '==' in str or '!=' in str
        # includes 'for'
        includes_for = 'for' in str
        # includes 'while'
        includes_while = 'while' in str
        # includes 'if'
        includes_if = 'if' in str
        # includes 'else'
        includes_else = 'else' in str
        # includes 'def'
        includes_def = 'def' in str
        # includes 'class'
        includes_class = 'class' in str
        # includes 'import'
        includes_import = 'import' in str
        # includes 'from'
        includes_from = 'from' in str
        # includes 'return'
        includes_return = 'return' in str
        # includes 'try'
        includes_try = 'try' in str
        # includes 'except'
        includes_except = 'except' in str
        # includes 'raise'
        includes_raise = 'raise' in str
        # includes 'pass'
        includes_pass = 'pass' in str
        # includes 'continue'
        includes_continue = 'continue' in str
        # includes 'break'
        includes_break = 'break' in str
        # includes 'assert'
        includes_assert = 'assert' in str
        # includes '''
        includes_quotes = '\'''' in str
        # concatenate all
        features = [numb_of_words, includes_quotes, includes_hash, includes_print, includes_equal, includes_for, includes_while, includes_if, includes_else, includes_def, includes_class, includes_import, includes_from, includes_return, includes_try, includes_except, includes_raise, includes_pass, includes_continue, includes_break, includes_assert]
        list_of_features.append(features)
    return list_of_features




def get_features(input_path, cudadevice, batchsize, include_embedding, output_path, maxusers, onlyAcceptReject):
    # load pickle file
    df_observations = pickle.load(open(input_path, 'rb'))
    global device, tokenizer, model
    device = torch.device('cuda:'+str(cudadevice) if torch.cuda.is_available() else 'cpu')

    if include_embedding:
        tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

        model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1").to(device)



    include_editpercentage = True
    include_timeinstate = True
    include_codelabels = True
    include_codeembeddings = include_embedding
    include_measurements = True
    include_userID = True
    include_textfeatures = True


    max_users =  min(maxusers, len(df_observations))

    df_observations_features = []
    label_to_enum = {'codeinit': 0, 'function def': 1, 'test_assert': 2, 'import': 3,
                    'control flow': 4, 'print': 5, 'error handling': 6, 'assignment': 7, 'comment': 8,
                    'binary_operator': 9, 'comparison': 10, 'expression': 11, 'docstring':12, 'other': 13}


    user_counter = 0
    feature_dict = {'Measurements: compCharLen, confidence, documentLength, numLines, numTokens, promptCharLen, promptEndPos, quantile': 0,
    'edit percentage': 1, 'time_in_state': 2, 'session_features':3, 'suggestion_label':4, 'prompt_label':5,
    'suggestion_embedding':6, 'prompt_embedding':7, 'suggestion_text_features':8, 'prompt_text_features':9,  'statename':10}
    for session in tqdm(df_observations):
        df_features = []
        logging.info(f'user {user_counter/len(df_observations)*100:.3f} \n \n' )
        if user_counter >= max_users:
            break
        user_counter += 1
        if len(session) == 0:
            continue
        session_features = []
        prev_row = [0] * 8
        # get prompt embedding
        indices_to_keep = []
        for i in range(len(session)):
            row = session.iloc[i]
            indices_to_keep.append(i)
        suggs_text = session.CurrentSuggestion.to_numpy()[indices_to_keep]
        prompts_text = session.CurrentPrompt.to_numpy()[indices_to_keep]
        # for each prompt only keep last 3 lines
        # split based on \n
        prompts_text = [prompt.split('\n') for prompt in prompts_text]
        prompts_text = [prompt[-3:] for prompt in prompts_text]
        # join back together

        prompts_text = ['\n'.join(prompt) for prompt in prompts_text]
        if include_codeembeddings:
            sugg_embedding = get_embedding_list(suggs_text)
            prompt_embedding = get_embedding_list(prompts_text)

        sugg_text_features = text_features(suggs_text)
        prompt_text_features = text_features(prompts_text)


        for i, index in enumerate(indices_to_keep):
            observation = []
            row = session.iloc[index]
            row_og = session.iloc[index]
            last_shown = copy.deepcopy(index)
            found_shown = False
            while not found_shown and last_shown >0:
                last_shown -= 1
                if session.iloc[last_shown]['StateName'] == 'Shown' or session.iloc[last_shown]['StateName'] == 'Replay':
                    found_shown = True
            if not found_shown:
                last_shown = max(0, index-1)
            if row_og['StateName'] != 'Accepted' and  row_og['StateName'] != 'Rejected':
                continue
            row = session.iloc[last_shown]



            try:
                # for Accepts and Rejects
                measurement_features = [row['Measurements']['compCharLen'],
                                    row['Measurements']['confidence'],
                                    row['Measurements']['documentLength'],
                                    row['Measurements']['numLines'],
                                    row['Measurements']['numTokens'],
                                    row['Measurements']['promptCharLen'],
                                    row['Measurements']['promptEndPos'],
                                    row['Measurements']['quantile'],
                                    row['Measurements']['meanAlternativeLogProb'],
                                    row['Measurements']['meanLogProb']]
                prev_row = measurement_features
            except:
                # for shown or browsing
                try:
                    measurement_features = [row['Measurements']['compCharLen'],
                                        prev_row[1],
                                        row['Measurements']['documentLength'],
                                        row['Measurements']['numLines'],
                                        row['Measurements']['numTokens'],
                                        row['Measurements']['promptCharLen'],
                                        row['Measurements']['promptEndPos'],
                                        prev_row[7],
                                        row['Measurements']['meanAlternativeLogProb'],
                                        row['Measurements']['meanLogProb']]
                except:
                    measurement_features = prev_row

            current_suggestion = row['CurrentSuggestion']
            # get embedding, get code feature

            # CurrentPrompt
            current_prompt = row['CurrentPrompt']
            # get last 5 lines of the prompt
            prompt_lines = current_prompt.split('\n')
            prompt_lines_last5 = prompt_lines[-1:]
            prompt_lines_last5_str = '\n'.join(prompt_lines_last5)

            lenght_sug = len(current_suggestion)
            lenght_prompt = len(current_prompt)
            lenght_sug_words = len(current_suggestion.split(' '))
            lenght_prompt_words = len(current_prompt.split(' '))
            new_measurements = [lenght_sug, lenght_prompt, lenght_sug_words, lenght_prompt_words, index]
            #measurement_features.extend(new_measurements)
            new_measurements.extend(measurement_features)

        

            edit_distance = row['EditPercentage']
            # CurrentSuggestion
            current_suggestion = row['CurrentSuggestion']
            # get embedding, get code feature

            # CurrentPrompt
            current_prompt = row['CurrentPrompt']
            # get last 5 lines of the prompt
            prompt_lines = current_prompt.split('\n')
            prompt_lines_last5 = prompt_lines[-1:]
            prompt_lines_last5_str = '\n'.join(prompt_lines_last5)


    
            time_spent_in_state = row['TimeSpentInState']

            if include_measurements:
                observation.append(new_measurements)

            if include_editpercentage:
                observation.append(edit_distance)

            if include_timeinstate:
                observation.append([time_spent_in_state])
                observation.append([index, index/len(session), len(session)])

            if include_codelabels:
                sugg_label = get_prompt_label(current_suggestion)
                sugg_label_enc = np.zeros(14) 
                sugg_label_enc[label_to_enum[sugg_label]] = 1
                prompt_label = get_prompt_label(prompt_lines[-1]) # label last line
                prompt_label_enc = np.zeros(14)
                prompt_label_enc[label_to_enum[prompt_label]] = 1
                observation.append(sugg_label_enc)
                observation.append(prompt_label_enc)

            if include_codeembeddings:
                observation.append(sugg_embedding[i])
                observation.append(prompt_embedding[i])
            else:
                observation.append(np.zeros(1))
                observation.append(np.zeros(1))
                
            if include_textfeatures:
                observation.append(np.array(sugg_text_features[i]))
                observation.append(np.array(prompt_text_features[i]))


            # add label
            observation.append(row_og['StateName'])

            # make observation into numeric np array
            observation = np.array(observation)#, dtype=np.float32)
            session_features.append(observation)

        df_observations_features.append(np.array(session_features))
        pickle.dump([df_observations_features, feature_dict, ], open(output_path, 'wb'))
    pickle.dump([df_observations_features, feature_dict, ], open(output_path, 'wb'))


def main():
    args = parser.parse_args()
    logging.info(args)
    if args.embedding not in [0,1]:
        raise ValueError('embedding argument must be 0 or 1')
    get_features(args.path, args.cudadevice, args.batchsize, args.embedding, args.output, args.maxusers, args.onlyacceptreject)

if __name__ == '__main__':
    main()

# call this script with
# python3 get_features.py --path ../data/observations.csv --cudadevice 0 --batchsize 32 --embedding True --output ../data/features.pkl --maxusers 100