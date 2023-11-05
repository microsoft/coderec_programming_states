# Understanding and Improving Copilot 

Code and Data for: 
- [Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming](https://arxiv.org/abs/2210.14306)
  [GO TO CODE](#Reading-Between-the-Lines)


- [When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming]( https://arxiv.org/pdf/2306.04930)
  [GO TO CODE](#When-to-Show-a-Suggestion)


Dataframe of telemetry for the study is available in [data pickle](data/data_labeled_study.pkl) and processed in folder [data](data/)

(Currently unavailable) Video data of the coding session is available as a zipped folder at https://drive.google.com/file/d/1qriGQXjMDoesr1WxB7s0QK8rYy2hSddc/view?usp=sharing 

# What is This?

- **Understanding how programmers use Copilot**: our work enables an in depth study of how programmers use Copilot, and the different states programmers are in when using Copilot (CUPS). This repository contributes telemetry data of programmers completing tasks with Copilot and a detailed analysis of that data.

- **When should a suggestion be shown in Copilot?** We contribute a simple method that predicts the probability of a programmer accepting a suggestion, and show that it can be used to improve the performance of Copilot by selectively showing suggestions.


# Installation

The environment named 'coderec' is available as a yml file and can be installed using:
```
conda env create -f environment.yml
```

Some of the packages are not required for all the scrips and notebooks, but are included in the environment for convenience. 

There is also a requirements.txt file available, but it does not include pip install so it is insufficient, to use that:
```
conda create --name coderec --file requirements.txt
```
We will also need to install further libraries and tools.

- Treesitter for getting code labels
```
 git clone https://github.com/tree-sitter/tree-sitter-python
```


#  Reading Between the Lines

All code is found [here](between_the_lines/)

## User Interface for Study Data

You can interact with the study data using our annotation interface.

Steps:

- First make sure to download the data including the labels (.json files) and the videos (.mp4 files). 
- Place both the json and mp4 in the 'user_study_webapp/app_study/static' folder.
- For each study,  run the following commands:

```
python server.py -p static/logs_user_8_l.json -v static/video_cropped_8.mp4
```
- Go to http://localhost:8080/ on your browser to see the interface.

Note that the jsons of the labeled states are not the final labels, please consult the [data pickle](data/data_labeled_study.pkl)

![Annotation Interface](images/interface_sreenshot.png)



We include the instructions for each coding task in [coding_tasks](between_the_lines/user_study_webapp/coding_tasks.ipynb)


## Drawing Timelines and Graphs


Use the jupyter notebook [viz_draw](between_the_lines/user_study_analysis/viz_draw.ipynb) to draw the timelines for the study data.

![User Timeline](images/user_timeline.PNG)



Use the jupyter notebook [viz_draw](between_the_lines/user_study_analysis/viz_draw.ipynb) to draw the graph for the study data.


![Graph](images/graph.JPG)


##  Analysis

For insights and analysis that are found in our paper, they can be replicated in the following  notebooks:

- [information and statistics about participants and sessions found in Section 5 of the paper](between_the_lines/user_study_analysis/section5.ipynb)
- [results of the user study found in Section 6 of the paper](between_the_lines/user_study_analysis/section6.ipynb)
- [predictive models of CUPS found in Section 7 of the paper](between_the_lines/predict_cups/predict_cups.ipynb)
- [recreate appendix graphs for post-study survey](between_the_lines//user_study_analysis/participants_analysis.ipynb)




## Dataframe for user study

Our study complete data is stored in '/data/data_labeled_study.pkl' which contains python  array which we will name 'df_observations' where: 'df_observations[i]' is the session for the ith user stored as a pandas dataframe.

To be more explicit, df_observations[i] is a pandas dataframe that contains the following columns:
```
'UserId' 
'TimeGenerated': timestamp for event
'StateName': betweeen 'Shown', 'Accepted', 'Rejected', 'Replay' (consecutive shown), 'Browsing' (shown different choice index)
'HiddenState' : high level hidden state between 'UserBeforeAction', 'UserPaused', 'UserTyping'
'TimeSpentInState'
'CurrentSuggestion' 
'CurrentPrompt'
'Measurements': measurements from logs
'EditPercentage': an array containing the relative edit distance (in 0-1) for the 5 stillincode events for this completion
'LabeledState': the state label by the user for the current state
```

The LabeledState takes the following values:
```
Thinking About Suggestion (a): actively thinking/verifying about suggestion shown, also includes going to the internet to verify <br>
Not Thinking (s): not thinking about suggestion shown  <br>
Deferring Thought For Later (d): decide to not think now about suggestion, but will think later about it  <br>
Thinking About New Code To Write (f): thinking about code outside suggestions to write, new functionality  <br>
Waiting For Suggestion (g): waiting for Copilot suggestion to be shown   <br>
Writing New Code (z): writing code that implements new functionality <br>
Editing Recent Suggestions (x): editing recent Copilot suggestions<br>
Editing (Personally) Written Code(c): editing code you wrote that was not a Copilot suggestion for purpose of fixing functionality <br>
Prompt Crafting (v): writing comment or code with intention of copilot completion<br>
Writing Documentation (b): adding comments for purpose of documentation,<br>
Debugging/Testing Code (h): running or debugging code to check functionality, may include writing tests or debugging statements<br>
Looking at documentation: looking online for documentation
```


# When to Show a Suggestion 

All code is found [here](when_to_show/)

![cdhf](images/cdhf.JPG)


## Generating Features from the Dataframe for Prediction and Decoding

We have already performed this step for you for the user study logs and processed data is in [data/featureframe_user_study.pkl](data/featureframe_user_study.pkl)

Given the extended logs, we will generate features for the prediction and decoding models.

The below command will generate a pickle file containing a python variable, name it 'df_features', of the following form:

df_features[i][j][h]: is the h'th feature for the k'th event for the ith user.

Let us elaborate further, df_features[i] is the all the data for the ith user. df_features[i][k] contains the features for the k'th event in the session. Finally, df_features[i][k][h] contains the h'th feature, more precisely, df_features[i][k] is a list of different feature, where df_features[i][k][h] is a list contains a representation of the h'th feature as follows:
```
   feature_dict = {'Measurements: compCharLen, confidence, documentLength, numLines, numTokens, promptCharLen, promptEndPos, quantile': 0,
    'edit percentage': 1, 'time_in_state': 2, 'session_features':3, 'suggestion_label':4, 'prompt_label':5,
    'suggestion_embedding':6, 'prompt_embedding':7, 'suggestion_text_features':8, 'prompt_text_features':9, 'statename':10}
```
meaning df_features[i][k][0] is a list contaning all measurement features, i.e. compCharLen, confidence, documentLength, numLines, numTokens, promptCharLen, promptEndPos, quantile in a row. And then df_features[i][k][6] is the 768 dimensional suggestion embedding and so forth.

The command to get the features pickle file is:
```
python action_prediction/generate_features.py -p'OUTPUT_PATH_EXTENDED_LOGS.pkl' \
-c 0 \
-b 1000 \
-o 'OUTPUT_PATH_features.pkl' \
-e 1 \
-m 99999  \ 
```
the documentation for the args is:
```
('-p', '--path', help='Path to extended logs frame', required=True) 
('-c', '--cudadevice', help='cuda device id', default=0, required=True, type=int)
('-b', '--batchsize', help='batch size', default=1000, required=True, type=int)
('-o', '--output', help='Output path of .pkl file', required=True) 
('-e', '--embedding', help='Whether to get embeddings for suggestion and prompt', required=True, type=int)
('-m', '--maxusers', help='max users', default=100, required=True, type=int)
```

## Replicating Figures and Tables

The figures and tables in the paper are generated using data from a larger dataset that is not provided for privacy reasons. However, one can perform all the analysis in the paper with the user study data that is collected in our 2022 paper. 

- Effect of Programmer Latent State analysis (page 6) can be replicated with the notebook [effect_of_latent_state.ipynb](./when_to_show/effect_of_latent_state.ipynb)

- The results (page 10) of predicting accepts with XGB can be replicated with [predict_accept_user_study.ipynb](./when_to_show/predict_accept_user_study.ipynb)

- Figure 7 (CDHF) can be replicated by first generating the models m1 and m2 and then running the notebook [cdhf.ipynb](./when_to_show/cdhf.ipynb)

- Figure 8 (which suggestion to show) and 10-11 can be replicated by first generating a model that only takes prompt and suggestion embeddings and running the notebook [max_reward_accept.ipynb](when_to_show/max_reward_accept.ipynb)

Other figures (Figure 5, Figure 6) require the larger dataset and are not included in this release.

# Citation

Please cite our papers if you use our dataset or code:

```
@article{mozannar2022reading,
  title={Reading Between the Lines: Modeling User Behavior and Costs in AI-Assisted Programming},
  author={Mozannar, Hussein and Bansal, Gagan and Fourney, Adam and Horvitz, Eric},
  journal={arXiv preprint arXiv:2210.14306},
  year={2022}
}

@article{mozannar2023show,
  title={When to Show a Suggestion? Integrating Human Feedback in AI-Assisted Programming},
  author={Mozannar, Hussein and Bansal, Gagan and Fourney, Adam and Horvitz, Eric},
  journal={arXiv preprint arXiv:2306.04930},
  year={2023}
}
```



# Other

## Acknowledgements
This release is part of research done during an internship at Microsoft Research ([privacy statement](https://privacy.microsoft.com/en-us/privacystatement)) and was based on valuable feedback from colleagues across MSR and GitHub including Saleema Amershi, Victor Dibia, Forough Poursabzi, Andrew Rice, Eirini Kalliamvakou, and Edward Aftandilian.



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
