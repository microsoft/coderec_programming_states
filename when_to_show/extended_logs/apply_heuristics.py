from click import edit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
from tqdm import tqdm
import pickle
import logging
logging.basicConfig(level=logging.INFO)
import  argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--logs', help='Path to extended state logs .pkl', required=True) # change to True
parser.add_argument('-o', '--output', help='Output path of .pkl file with heuristic column', required=True) # change to True

# this script takes output of extended_logs_mass and adds a column which is Heuristic

'''
Thinking About Suggestion (a): actively thinking/verifying about suggestion shown, also includes going to the internet to verify <br>
Not Thinking (s): not thinking about suggestion shown  <br>
Deferring Thought For Later (d): decide to not think now about suggestion, but will think later about it  <br>
Thinking About New Code To Write (f): thinking about code outside suggestions to write, new functionality  <br>
Waiting For Suggestion (g): waiting for Copilot suggestion to be shown   <br>
Writing New Code (z): writing code that implements new functionality <br>
Edditing Recent Suggestions (x): editing recent Copilot suggestions<br>
Editing (Personally) Written Code(c): editing code you wrote that was not a Copilot suggestion for purpose of fixing functionality <br>
Prompt Crafting (v): writing comment or code with intention of copilot completion<br>
Writing Documentation (b): adding comments for purpose of documentation,<br>
Debugging/Testing Code (h): running or debugging code to check functionality, may include writing tests or debugging statements<br>
Looking at documentation: looking online for documentation
IDK (n): unsure what you were doing or which state fits the label, can also write custom state
'''


# we will define one step heuristics first
# we define heuristics as a function of a span of logs, or a single entry
class Heuristics:
    def __init__(self, observed_states = None, unobserved_decoding = None):
        self.observed_states = ['UserBeforeAction', 'UserTyping', 'UserPaused', 'UserTypingOrPaused']
        self.unobserved_decoding = ['ThinkingSuggestion', 'NoThought', 'DeferringThought',  'ThinkingNewCode',
                                 'WaitingSuggestion', 'WritingNewCode', 'EditingSuggestions', 'Editing',
                                  'PromptCrafting', 'WritingDocumentation', 'DebuggingCode', 'LookingAtDocumentation']#, 'IDK']

        self.unobserved_decoding_dict = {'ThinkingSuggestion': 0, 'NoThought': 1, 'DeferringThought': 2,  'ThinkingNewCode': 3,
                                 'WaitingSuggestion': 4, 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing': 7,
                                  'PromptCrafting': 8, 'WritingDocumentation': 9, 'DebuggingCode': 10, 'LookingAtDocumentation': 11}#, 'IDK': 11}

    def label_state(self, logs_session, index):
        # function documentation input or output
        # logs_session: dataframe of logs for a single session
        # index: index of the current state
        # return: distribution over hidden state

        uniform_state = np.ones(len(self.unobserved_decoding))/len(self.unobserved_decoding)
        all_labels = []
        for function in dir(self):
            if function.startswith('heuristic'):
                label = getattr(self, function)(logs_session, index)
                if (label is not None):
                    all_labels.append(label)
        final_label = []
        if len(all_labels) >0:
            final_label = np.mean(all_labels, axis = 0)
            final_label = final_label/np.sum(final_label)
        else:
            if logs_session.iloc[index]['HiddenState'] == 'UserBeforeAction':
                final_label = np.zeros(len(self.unobserved_decoding))
                final_label[self.unobserved_decoding_dict['ThinkingSuggestion']] = 1
                final_label[self.unobserved_decoding_dict['NoThought']] = 1
                final_label[self.unobserved_decoding_dict['DeferringThought']] =1 
                final_label[self.unobserved_decoding_dict['ThinkingNewCode']] = 1
                final_label = final_label + uniform_state
                final_label = final_label/np.sum(final_label)
            elif logs_session.iloc[index]['HiddenState'] == 'UserTyping':
                final_label = np.zeros(len(self.unobserved_decoding))
                final_label[self.unobserved_decoding_dict['WritingNewCode']] = 1
                final_label[self.unobserved_decoding_dict['EditingSuggestions']] = 1
                final_label[self.unobserved_decoding_dict['Editing']] =1 
                final_label = final_label + uniform_state
                final_label = final_label/np.sum(final_label)
            elif logs_session.iloc[index]['HiddenState'] == 'UserPaused':
                final_label = np.zeros(len(self.unobserved_decoding))
                final_label[self.unobserved_decoding_dict['ThinkingNewCode']] = 1
                final_label[self.unobserved_decoding_dict['WaitingSuggestion']] = 1
                final_label[self.unobserved_decoding_dict['NoThought']] =1 
                final_label = final_label + uniform_state
                final_label = final_label/np.sum(final_label)
            else:
                final_label = uniform_state
        return final_label
        
    
    def heuristic_nothought(self, logs_session, index):
        
        try:
            row = logs_session.iloc[index]
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            #next_row = logs_session.iloc[index+1]
            # no thought: single line rejected 
            if row['HiddenState'] != 'UserBeforeAction':
                return None

            if row["TimeSpentInState"] <= 0.1:# and \
            #row["Measurements"]["compCharLen"] <= 20 and \
            #row["Measurements"]["numLines"] == 1 and \
            #row["StateName"] in ['Rejected','Browsing']:
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.8
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.1
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.05
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.05
                return label_heuristic
            #next_row = logs_session.iloc[index+1]
            # no thought: multi line accepted in little time and not edited later

            if row["TimeSpentInState"] <= 0.3 and \
            row["Measurements"]["numLines"] > 2 and \
            row["StateName"] == 'Accepted' and \
            row['EditPercentage'][0] <= 0.1:          
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.5
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.2
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.1
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.2
                return label_heuristic
    
            if row["TimeSpentInState"] > 100 and \
            row["StateName"] == 'Accepted':
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.5
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.05
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.25
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.2
                return label_heuristic
            return None
        except:
            return None

    
    def heuristic_thinking(self, logs_session, index):
        # thinking about suggestion, significant time spent looking, and suggestion was not edited in the future and accepted.
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserBeforeAction':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            #next_row = logs_session.iloc[index+1]
            # no thought: single line rejected 
            row = logs_session.iloc[index]
            next_row = logs_session.iloc[index+1]   
            if row["TimeSpentInState"] >= 3 and \
                row["StateName"] == 'Accepted':
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.0
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.3
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.6
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.1

                return label_heuristic

        # thinking about suggestion, significant time spent looking, then rejected, and then typed next state
            if row["TimeSpentInState"] >= 5 and \
                row["StateName"] == 'Rejected' and \
                next_row["HiddenState"] in ['UserTyping']:
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.33
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.34
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.33

                return label_heuristic
            return None
        except:
            return None



    def heuristic_defer(self, logs_session, index):
        # thinking about suggestion, significant time spent looking, and suggestion was not edited in the future and accepted.
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserBeforeAction':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            #next_row = logs_session.iloc[index+1]
            # no thought: single line rejected 
            row = logs_session.iloc[index]
            if row["TimeSpentInState"] <= 1 and \
                row["Measurements"]["numLines"] > 2 and \
                row["StateName"] == 'Accepted':# and \
                #row['EditPercentage']['relativeLexEditDistance'] > 0.3: 
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.3
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.35
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.3
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.05
                return label_heuristic

            prev_rows_states = logs_session.iloc[index-4:index].StateName.to_numpy() # -3 and -7 should be accepts
            prev_rows_hidden_states = logs_session.iloc[index-4:index].HiddenState.to_numpy() # -3 and -7 should be accepts
            row = logs_session.iloc[index]
            next_row = logs_session.iloc[index+1]
            prev_state_condition =  ['Accepted', 'Shown', 'Accepted', 'Shown']
            prev_hidden_state_condition =  ['UserBeforeAction', 'UserPaused', 'UserBeforeAction', 'UserPaused']
            prev_hidden_state_check = True
            prev_state_check = True
            # enumerate through prev_rows_states
            counter = 0
            for i in range(len(prev_state_condition)):
                if prev_rows_states[i] != prev_state_condition[counter]:
                    prev_state_check = False
                    break
                if prev_hidden_state_condition[i] == 'UserPaused' and prev_rows_hidden_states[i] != prev_hidden_state_condition[counter]:
                    prev_hidden_state_check = False
                    break
                counter += 1
            if row["TimeSpentInState"] <= 1 and \
            row["Measurements"]["numLines"] == 1 and \
            row["StateName"] == 'Accepted' and \
            prev_state_check and prev_hidden_state_check:
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.1
                label_heuristic[self.unobserved_decoding_dict['DeferringThought']] = 0.7
                label_heuristic[self.unobserved_decoding_dict['ThinkingSuggestion']] = 0.1
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.1

                return label_heuristic
            return None
        except:
            return None


    def heuristic_thinkingnew(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserBeforeAction':# or row['HiddenState'] != 'UserPaused':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            next_row = logs_session.iloc[index+1]
            if row["TimeSpentInState"] > 3 and \
                row["StateName"] in ['Rejected','Replay','Browsing']: #and \
                #next_row["HiddenState"] in ['UserTyping']:
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.25
                label_heuristic[self.unobserved_decoding_dict['WaitingSuggestion']] = 0.0
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.75
                return label_heuristic

        except:
            return None

    def heuristic_waiting(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserPaused':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            next_row = logs_session.iloc[index+1]
            next_next_row = logs_session.iloc[index+2]
            if row["TimeSpentInState"] > 3 and \
                next_next_row["StateName"] in ['Accepted']: #and \
                #next_row["HiddenState"] in ['UserTyping']:
                label_heuristic[self.unobserved_decoding_dict['NoThought']] = 0.0
                label_heuristic[self.unobserved_decoding_dict['WaitingSuggestion']] = 0.55
                label_heuristic[self.unobserved_decoding_dict['ThinkingNewCode']] = 0.45
                return label_heuristic

        except:
            return None


    def heuristic_writingnew(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserTyping':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            if row["TimeSpentInState"] > 1 and \
                row['EditPercentage'][0] == 0:
               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['WritingNewCode']] = 0.4
                label_heuristic[self.unobserved_decoding_dict['EditingSuggestions']] = 0.2
                label_heuristic[self.unobserved_decoding_dict['Editing']] = 0.4
                return label_heuristic

        except:
            return None

    def heuristic_editingsuggestions(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserTyping':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            if row["TimeSpentInState"] > 0.5 and \
                (row['EditPercentage'][0] > 0 or row['EditPercentage'][1] > 0) :
               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['WritingNewCode']] = 0.2
                label_heuristic[self.unobserved_decoding_dict['EditingSuggestions']] = 0.6
                label_heuristic[self.unobserved_decoding_dict['Editing']] = 0.2
                return label_heuristic
            if row["TimeSpentInState"] < 0.5 and \
                (row['EditPercentage'][0] > 0 or row['EditPercentage'][1] > 0) :
               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['WritingNewCode']] = 0.0
                label_heuristic[self.unobserved_decoding_dict['EditingSuggestions']] = 0.6
                label_heuristic[self.unobserved_decoding_dict['Editing']] = 0.4
                return label_heuristic


        except:
            return None

    def heuristic_editing(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserTyping':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            if row["TimeSpentInState"] > 1 and \
                row['EditPercentage'][0] == 0:
               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['WritingNewCode']] = 0.4
                label_heuristic[self.unobserved_decoding_dict['EditingSuggestions']] = 0.2
                label_heuristic[self.unobserved_decoding_dict['Editing']] = 0.4
                return label_heuristic

        except:
            return None

    # 'PromptCrafting', 'WritingDocumentation', 'DebuggingCode', 'LookingAtDocumentation'
    def heuristic_promptcrafting(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserTyping' or row['StateName'] != 'UserPaused':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            # get current prompt
            prompt = row['CurrentPrompt']
            # get last line in prompt
            prompt_lines = prompt.split('\n')
            last_line = prompt_lines[-1]
            # check if last_line starts with #
            last_prompt_is_comment = last_line.startswith('#')

            if last_prompt_is_comment:
                label_heuristic[self.unobserved_decoding_dict['PromptCrafting']] = 0.5
                label_heuristic[self.unobserved_decoding_dict['WritingDocumentation']] = 0.5
                return label_heuristic

        except:
            return None

    def heuristic_writingdoc(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            if row['HiddenState'] != 'UserTyping' or row['StateName'] != 'UserPaused':
                return None
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            # get current prompt
            prompt = row['CurrentPrompt']
            # get last line in prompt
            prompt_lines = prompt.split('\n')
            last_line = prompt_lines[-1]
            # check if last_line starts with #
            last_prompt_is_comment = last_line.startswith('#')

            if last_prompt_is_comment:
                label_heuristic[self.unobserved_decoding_dict['PromptCrafting']] = 0.5
                label_heuristic[self.unobserved_decoding_dict['WritingDocumentation']] = 0.5
                return label_heuristic

        except:
            return None

    def heuristic_DebuggingCode(self, logs_session, index):
        try:
            if row['HiddenState'] != 'UserTyping':
                return None
            row = logs_session.iloc[index]
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            if row["TimeSpentInState"] > 50:               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['DebuggingCode']] = 0.55
                label_heuristic[self.unobserved_decoding_dict['LookingAtDocumentation']] = 0.45
                return label_heuristic

        except:
            return None

    def heuristic_LookingAtDocumentation(self, logs_session, index):
        try:
            row = logs_session.iloc[index]
            label_heuristic = np.zeros(len(self.unobserved_decoding))
            if row["TimeSpentInState"] > 50:               #next_row["HiddenState"] in ['UserTyping']:
               # 'WritingNewCode': 5, 'EditingSuggestions': 6, 'Editing'
                label_heuristic[self.unobserved_decoding_dict['DebuggingCode']] = 0.5
                label_heuristic[self.unobserved_decoding_dict['LookingAtDocumentation']] = 0.5
                return label_heuristic

        except:
            return None

# Apply Heuristics to the dataframe



def main():

    args = parser.parse_args()
    # pickle load
    df_observations = pickle.load(open(args.logs, 'rb'))
    heuristics_uba = Heuristics()
    heuristic_coverage = 0
    heuristic_could_apply = 0
    logging.info('Multiple Users, starting heuristics')
    try:
        test_a = df_observations[0][0].iloc[0:1]
    except:
        df_observations = [df_observations]
    for user_sessions_idx in tqdm(range(len(df_observations))):
        for session_id in range(len(df_observations[user_sessions_idx])):
            # iterate through dataframe and modify statename
            heuristic_labels_session = []
            if len(df_observations[user_sessions_idx][session_id]) > 0:
                for index in range(len(df_observations[user_sessions_idx][session_id])):
                    heuristic_could_apply += 1
                    heuristic_label = heuristics_uba.label_state(df_observations[user_sessions_idx][session_id], index)
                    # check if all elements of heuristic_label are equal
                    if not all(x == heuristic_label[0] for x in heuristic_label):
                        heuristic_coverage += 1
                    heuristic_labels_session.append(heuristic_label)
            # add Heuristic column
            df_observations[user_sessions_idx][session_id]['HeuristicHiddenState'] = heuristic_labels_session
    # pickle dump
    pickle.dump(df_observations, open(args.output, 'wb'))
    logging.info('Successfully applied heuristics to')
    
main()