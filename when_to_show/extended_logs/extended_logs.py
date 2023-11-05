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
parser.add_argument('-p', '--nonrestricted', help='Path to non restricted logs csv ', required=True) # change to True
parser.add_argument('-r', '--restricted', help='Path to non restricted logs csv', required=True) # change to True
parser.add_argument('-o', '--output', help='Output path of .pkl file', required=True) # change to True


def retreive_logs(logs_path, restricted_logs_path):
    df_logs = pd.read_csv (logs_path)
    df_logs_restr = pd.read_csv(restricted_logs_path)
    # convert json into dict
    df_logs.PropertiesJson = [json.loads(df_logs.PropertiesJson[i]) for i in range(len(df_logs.PropertiesJson))]
    df_logs.MeasurementsJson = [json.loads(df_logs.MeasurementsJson[i]) for i in range(len(df_logs.MeasurementsJson))]
    # mark new sessions based on timestamp difference of more than 30 minutes
    timestamps = df_logs.TimeGenerated.to_numpy()
    timestamps_datetime = [pd.to_datetime(timestamps[i]) for i in range(len(timestamps))]
    #  difference between timestamps
    timestamps_diff = [timestamps_datetime[i+1] - timestamps_datetime[i] for i in range(len(timestamps_datetime)-1)]
    #  convert to minutes
    timestamps_diff_min = [timestamps_diff[i].total_seconds()/60 for i in range(len(timestamps_diff))]
    # check if consecutive differences are bigger than 30 minutes 
    timestamps_diff_min_consecutive = [1] + [1 if timestamps_diff_min[i-1] > 30 else 0 for i in range(1,len(timestamps_diff_min)+1)] 
    # create array where if timestamps_diff_min_consecutive is 1 you add 1 to the previous value
    new_session_counter = [0] * len(timestamps_diff_min_consecutive)
    for i in range(1, len(timestamps_diff_min_consecutive)):
        new_session_counter[i] = new_session_counter[i-1] + timestamps_diff_min_consecutive[i]


    df_logs['newSession'] = timestamps_diff_min_consecutive
    df_logs['SessionID'] = new_session_counter

    df_logs_restr.PropertiesJson = [json.loads(df_logs_restr.PropertiesJson[i]) for i in range(len(df_logs_restr.PropertiesJson))]
    df_logs_restr.MeasurementsJson = [json.loads(df_logs_restr.MeasurementsJson[i]) for i in range(len(df_logs_restr.MeasurementsJson))]

    return df_logs, df_logs_restr

# create observable state diagram
def get_extended_states(df_logs, df_logs_restr, extented_model_path):
    column_names = ["TimeGenerated", "CompletionId", "UserId", "SessionId","StateName", "HiddenState","TimeSpentInState","CurrentSuggestion", "CurrentPrompt", "Measurements"]
    unique_session_ids = np.unique(df_logs.SessionID) # or SessionId  
    df_observations_states = []
    for id_unique in tqdm(unique_session_ids):
        # split on SessionID
        df_task = df_logs[df_logs.SessionID == id_unique]
        df_names = df_task.Name.to_numpy()
        df_states = pd.DataFrame(columns= column_names)
        in_Shown = False
        last_completion_id = 0 
        # iterate through data frame
        last_timestamp = -1
        for index, row in df_task.iterrows():
            # ignore stillInCodes for now
            name = row['Name']
            completion_id = row['CompletionId']
            if name == 'copilot/ghostText.stillInCode':
                continue
            choiceIndex = row.PropertiesJson['choiceIndex']
            df_logs_restr_completion = df_logs_restr[df_logs_restr.CompletionId == completion_id]
            completion = ""
            prompt = ""
            # iterate through restricted data frame to get prompt and completion
            for index_restr, row_restr in df_logs_restr_completion.iterrows():
                if 'completionTextJson' in row_restr.PropertiesJson and 'choiceIndex' in row_restr.PropertiesJson:
                    if row_restr.PropertiesJson['choiceIndex'] == choiceIndex:
                        completion = row_restr.PropertiesJson['completionTextJson']
                if 'hypotheticalPromptJson' in row_restr.PropertiesJson:
                    prompt = row_restr.PropertiesJson['hypotheticalPromptJson']

            # find prompt in completion 
            index_restr = df_logs_restr_completion.index[0]
            df_prompt_compl = df_logs_restr.loc[index_restr-1]
            if 'promptJson' in df_prompt_compl.PropertiesJson:
                prompt = df_prompt_compl.PropertiesJson['promptJson']

            user_id = row['UserId']
            session_id = row['SessionId']

            # just keep track if in shown state
            if name != 'copilot/ghostText.shown' and name != 'copilot/ghostText.shownFromCache':
                in_Shown = False
                last_completion_id = completion_id

            if last_timestamp == row['TimeGenerated']:
                continue

            if last_timestamp != -1:
                # difference
                time_difference =  pd.to_datetime(row['TimeGenerated']) - pd.to_datetime(last_timestamp) 
                time_difference_sec = time_difference.total_seconds()
            else:
                last_timestamp = row['TimeGenerated']
                time_difference_sec = 0 
            

            state_dict = {  'TimeGenerated': row['TimeGenerated'],
                            'CompletionId': completion_id,
                            'UserId': user_id,
                            'SessionId': session_id,
                            'StateName': "TBD",
                            'HiddenState': "TBD",
                            'TimeSpentInState': time_difference_sec,
                            'CurrentSuggestion': completion,
                            'CurrentPrompt': prompt,
                            'Measurements': [row['MeasurementsJson']]} # might add more properties later
            


            # all states tracking
            if name == 'copilot/ghostText.shown' or name == 'copilot/ghostText.shownFromCache':
                # if previous state was also shown
                if in_Shown:
                    # if we are in shown and the completion id is the same as the last one, user is browsing suggestions
                    if last_completion_id == completion_id:
                        new_state = copy.deepcopy(state_dict)
                        new_state['StateName'] = 'Browsing'
                        new_state['HiddenState'] = 'UserBeforeAction'

                        df_states = pd.concat([df_states, pd.DataFrame(new_state)])

                    # if we were previously in shown, and now we have a new shown, then we previously rejected the shown
                    # typed, and then got another suggestion
                    else:
                        new_state = copy.deepcopy(state_dict)
                        new_state['StateName'] = 'Shown' # hidden rejected
                        new_state['HiddenState'] = 'UserTypingOrPaused'

                        df_states = pd.concat([df_states, pd.DataFrame(new_state)])
                else:
                    # if previous was not shown, then user was typing/paused
                    new_state = copy.deepcopy(state_dict)
                    new_state['StateName'] = 'Shown'
                    new_state['HiddenState'] = 'UserTypingOrPaused'
                    
                    df_states = pd.concat([df_states, pd.DataFrame(new_state)])          
                last_completion_id = completion_id
                in_Shown = True

            elif name == 'copilot/ghostText.accepted':
                # before accepting, user was thinking
                new_state = copy.deepcopy(state_dict)
                new_state['StateName'] = 'Accepted'
                new_state['HiddenState'] = 'UserBeforeAction'
                df_states = pd.concat([df_states, pd.DataFrame(new_state)])
            elif name == 'copilot/ghostText.rejected':
                new_state = copy.deepcopy(state_dict)
                new_state['StateName'] = 'Rejected'
                new_state['HiddenState'] = 'UserBeforeAction'
                df_states = pd.concat([df_states, pd.DataFrame(new_state)])
            last_timestamp = row['TimeGenerated']

        df_observations_states.append(df_states)
    logging.info("finished adding hidden states and times")
    # decipher paused vs typing
    typing_count = 0
    paused_count = 0
    for session_id in range(len(df_observations_states)):
        # iterate through dataframe and modify statename
        for index in range(1, len(df_observations_states[session_id])):
            if df_observations_states[session_id].iloc[index]['StateName'] == 'Shown':

                if df_observations_states[session_id].iloc[index-1]['StateName'] == 'Accepted':
                    code_lenght = df_observations_states[session_id].iloc[index]['Measurements']['documentLength']
                    index2 = index -1
                    code_lenght2 = df_observations_states[session_id].iloc[index2]['Measurements']['documentLength']
                    suggestion_lenght = df_observations_states[session_id].iloc[index2]['Measurements']['compCharLen']
                    if abs(code_lenght2 + suggestion_lenght - code_lenght)<=3:
                        df_observations_states[session_id].iloc[index,df_observations_states[session_id].columns.get_loc('HiddenState')]= 'UserPaused'
                        paused_count += 1
                        
                    else:
                        df_observations_states[session_id].iloc[index,df_observations_states[session_id].columns.get_loc('HiddenState')]= 'UserTyping'
                        typing_count += 1
                        
                else:
                    code_lenght = df_observations_states[session_id].iloc[index]['Measurements']['documentLength']
                    index2 = index -1
                    code_lenght2 = df_observations_states[session_id].iloc[index2]['Measurements']['documentLength']
                    if abs(code_lenght2  - code_lenght)<=2:
                        df_observations_states[session_id].iloc[index,df_observations_states[session_id].columns.get_loc('HiddenState')]= 'UserPaused'
                        paused_count += 1                        
                        
                    else:
                        df_observations_states[session_id].iloc[index,df_observations_states[session_id].columns.get_loc('HiddenState')]= 'UserTyping'
                        typing_count += 1
                        
    logging.info(f'paused_count: {paused_count} typing count: {typing_count}' )
    logging.info("finished  decyphering typing/paused")

    # add editDistance
    for session_id in range(len(df_observations_states)):
        EditPercentage = []
        # iterate through dataframe and modify statename
        for index in range(len(df_observations_states[session_id])):
            # get completion id
            completion_id = df_observations_states[session_id].iloc[index]['CompletionId']
            # get all logs in df_logs that have matching completion_id
            df_task_completion = df_logs[df_logs.CompletionId == completion_id]
            df_task_completion  = df_task_completion[df_task_completion.Name == 'copilot/ghostText.stillInCode']
            edit_distance = {'charEditDistance': -1, 'lexEditDistance': -1, 'stillInCodeHeuristic': -1, 'relativeLexEditDistance': 1}
            # iterate through df_task_completion
            for index2 in range(len(df_task_completion)):
                # get measurements
                edit_distance['charEditDistance'] = df_task_completion.iloc[index2]['MeasurementsJson']['charEditDistance']
                edit_distance['lexEditDistance'] = df_task_completion.iloc[index2]['MeasurementsJson']['lexEditDistance']
                edit_distance['stillInCodeHeuristic'] = df_task_completion.iloc[index2]['MeasurementsJson']['stillInCodeHeuristic']
                edit_distance['relativeLexEditDistance'] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']
                if df_task_completion.iloc[index2]['MeasurementsJson']['timeout'] == 600.0:
                    break

            EditPercentage.append(edit_distance)
        df_observations_states[session_id]['EditPercentage'] = EditPercentage


    pickle.dump(df_observations_states, open(extented_model_path, 'wb'))
    logging.info("saved logs into " + extented_model_path)


def main():
    args = parser.parse_args()
    df_logs, df_logs_restr = retreive_logs(args.nonrestricted, args.restricted)
    get_extended_states(df_logs, df_logs_restr, args.output)

if __name__ == '__main__':
    main()

# python extended_logs.py -p '../data/logs_7-13_hussein.csv' -r '../data/restricted_7-13_hussein.csv' -o '../data/generated_data/extended_states_7-13.pkl'