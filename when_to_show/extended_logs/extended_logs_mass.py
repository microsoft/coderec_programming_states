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
parser.add_argument('-p', '--folder', help='Path to folder with (non) and restricted logs csvs', required=True) # change to True
parser.add_argument('-o', '--output', help='Output path of .pkl file', required=True) # change to True


# create observable state diagram
def get_extended_states(df_logs, df_logs_restr):
    column_names = ["TimeGenerated", "CompletionId", "TrackingId", "SessionId","StateName", "HiddenState","TimeSpentInState","CurrentSuggestion", "CurrentPrompt", "Measurements"]
    unique_session_ids = np.unique(df_logs.SessionID) # or SessionId  
    df_observations_states = []
    for id_unique in tqdm(unique_session_ids):
        # split on SessionID
        df_task = df_logs[df_logs.SessionID == id_unique]
        df_names = df_task.Name.to_numpy()
        df_states = pd.DataFrame(columns= column_names)
        in_Shown = False
        last_completion_id = 0 
        last_choiceindex = -1
        # iterate through data frame
        last_timestamp = -1
        for index, row in df_task.iterrows():
            # ignore stillInCodes for now
            name = row['Name']
            completion_id = row['CompletionId']
            if name == 'copilot/ghostText.stillInCode':
                continue
            
            choiceIndex = row.choiceIndex
            df_logs_restr_completion = df_logs_restr[df_logs_restr.CompletionId == completion_id]
            completion = ""
            prompt = ""
            # iterate through restricted data frame to get prompt and completion
            for index_restr, row_restr in df_logs_restr_completion.iterrows():
                if row_restr.completionTextJson != "" and pd.notna(row_restr.completionTextJson):
                    if row_restr.choiceIndex == choiceIndex:
                        completion = row_restr.completionTextJson
                if row_restr.PromptJson != "" and pd.notna(row_restr.PromptJson):
                    prompt = row_restr.PromptJson
                if prompt == "" and  row_restr.hypotheticalPromptJson != "" and pd.notna(row_restr.hypotheticalPromptJson):
                    prompt = row_restr.hypotheticalPromptJson
                

            # find prompt in completion 

            user_id = row['TrackingId']
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
                            'TrackingId': user_id,
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
                        if last_choiceindex != choiceIndex:
                            new_state['StateName'] = 'Browsing'
                        else:
                            new_state['StateName'] = 'Replay'
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
            last_choiceindex = choiceIndex
        df_observations_states.append(df_states)
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
            edit_distance = [-1]*5 # 15,30, 120, 300, 600 seconds, only for relativedistance
            edit_distance_keys = {'charEditDistance': -1, 'lexEditDistance': -1, 'stillInCodeHeuristic': -1, 'relativeLexEditDistance': 1}
            # iterate through df_task_completion

            for index2 in range(len(df_task_completion)):
                # get measurements
                timeout = int(df_task_completion.iloc[index2]['MeasurementsJson']['timeout'])
                if timeout == 15:
                    edit_distance[0] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']
                elif timeout == 30:
                    edit_distance[1] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']
                elif timeout == 120:
                    edit_distance[2] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']
                elif timeout == 300:
                    edit_distance[3] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']
                elif timeout == 600:
                    edit_distance[4] = df_task_completion.iloc[index2]['MeasurementsJson']['relativeLexEditDistance']


            for j in range(len(edit_distance)):
                if j == 0 and edit_distance[j] == -1:
                    edit_distance[j] = 0
                elif edit_distance[j] == -1:
                    edit_distance[j] = edit_distance[j-1]
            EditPercentage.append(edit_distance)
        df_observations_states[session_id]['EditPercentage'] = EditPercentage


    return df_observations_states


def get_df_all(folder_path, output_path):
    # check all files in a folder
    def get_file_paths(folder_path):
        import os
        files = os.listdir(folder_path)
        return files

    file_paths = get_file_paths(folder_path)
    df_logs_nonr = []
    df_logs_r = []
    for file in file_paths:
        # check if file is csv with ccontent
        if not file.endswith('.csv'):
            continue

        if file[0] == "r":
            # restricted
            df_logs = pd.read_csv (folder_path +"/"+file)
            df_logs.MeasurementsJson = [json.loads(df_logs.MeasurementsJson[i]) for i in range(len(df_logs.MeasurementsJson))]
            df_logs_r.append(df_logs)
        else:
            # non-restricted
            df_logs = pd.read_csv (folder_path +"/"+file)
            df_logs.MeasurementsJson = [json.loads(df_logs.MeasurementsJson[i]) for i in range(len(df_logs.MeasurementsJson))]
            df_logs_nonr.append(df_logs)

    # merge pandas dataframes
    df_logs_nonr = pd.concat(df_logs_nonr)
    df_logs_r = pd.concat(df_logs_r)

    # split dataframe based on UserId into a dictionary
    def split_df_by_user(df):
        df_by_user = {}
        for user in df.TrackingId.unique():
            df_by_user[user] = df[df.TrackingId == user]
        return df_by_user
    split_df_by_user_nonr = split_df_by_user(df_logs_nonr)
    split_df_by_user_r = split_df_by_user(df_logs_r)

    for user in split_df_by_user_nonr:
        timestamps = split_df_by_user_nonr[user].TimeGenerated.to_numpy()
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
        split_df_by_user_nonr[user]['newSession'] = timestamps_diff_min_consecutive
        split_df_by_user_nonr[user]['SessionID'] = new_session_counter


    user_ids = split_df_by_user_nonr.keys()
    user_ids_r = split_df_by_user_r.keys()
    # check overlap between non-restricted and restricted dataframes
    user_ids_overlap = [user for user in user_ids if user in user_ids_r]

    df_observations_states_users = []
    for user in user_ids_overlap:
        observations_user = get_extended_states(split_df_by_user_nonr[user], split_df_by_user_r[user])
        df_observations_states_users.append(observations_user) # extend if not by user
        logging.info(f'finished user: {user}')
        logging.info('\n')
    # sum lenght of all users
    logging.info(f'Total length of all sessions: {sum([len(df) for df in df_observations_states_users])}')
    pickle.dump(df_observations_states_users, open(output_path, 'wb'))

def main():
    '''
    returns: list sessions
    sessions[i] is a list of dataframes for user i, meaning sessions[i][j] is the jth dataframe (pd dataframe) for user i
    '''
    args = parser.parse_args()
    get_df_all(args.folder, args.output)

if __name__ == '__main__':
    main()

# python extended_logs_mass.py -p "../data/mass_data" -o "../data/generated_data/mass_logs_extended.pkl"