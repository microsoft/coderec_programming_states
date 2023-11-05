
# Extracting structured dataframes from raw Copilot logs

(this is only useful if you have access to raw telemetry logs and want to convert to the format that we already have done.)


extended_logs_mass : goes from raw Copilot logs (both restricted and non-restricted) to a list of dataframes for each user in the logs. It takes in a folder which contains restricted and non restricted logs for all users.

It stores in a pickle file an array which we will name 'df_observations' where: 'df_observations[i][j]' contains the jth observation of the ith user stored as a pandas dataframe with the following columns:
```
"TimeGenerated",
"CompletionId",
"TrackingId",
"SessionId",
"StateName",
"HiddenState",
"TimeSpentInState",
"CurrentSuggestion",
"CurrentPrompt",
"Measurements"
```

extended_logs: same as the above but for only one user and takes in restricted logs csv and non_restricted logs csv.





! python apply_heuristics.py -p '../data/generated_data/heuristics_test.pkl' -o '../data/generated_data/heuristics_test.pkl'