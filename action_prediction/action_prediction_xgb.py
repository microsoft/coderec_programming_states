
from requests import session
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy
import numpy as np, scipy.stats as st
import json
from tqdm import tqdm
import pickle
import logging
import  argparse
import math
from xgboost import XGBClassifier
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from itertools import chain
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
logging.basicConfig(level=logging.INFO)
from sklearn import preprocessing
# two local imports
from action_prediction_prep import get_features_labels
from action_prediction_prep import process_data
import matplotlib
# plot calibration of Copilot confidences with XGBoost predictions
from re import S
from scipy.stats.stats import pearsonr  
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import  argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='Path to features array', required=True) # change to True
parser.add_argument('-c', '--usegpu', help='to use gpu (0 or 1)', default=0, required=True, type=int)
parser.add_argument('-s', '--splitbyusers', help='split by users or session (1 or 0)', default=0, required=True, type=int)
parser.add_argument('-o', '--output', help='output path folder',  required=True)
parser.add_argument('-t', '--testpercentage', help='test percentage', default = 0.2, type =float)
parser.add_argument('-v', '--valpercentage', help='val percentage', default =0.1, type=float)




def main():
    args = parser.parse_args()
    path = args.path
    use_gpu = args.usegpu
    splitbyusers = args.splitbyusers
    output_path = args.output
    test_percentage = args.testpercentage
    val_percentage = args.valpercentage
    REMOVE_S_AND_R = True # remove shown and replay
    features_to_keep = np.array([0,4,5,6,7,8,9])
    label_index = np.array([10])
    feature_dict = {'Measurements: compCharLen, confidence, documentLength, numLines, numTokens, promptCharLen, promptEndPos, quantile': 0,
    'edit percentage': 1, 'time_in_state': 2, 'session_features':3, 'suggestion_label':4, 'prompt_label':5,
    'suggestion_embedding':6, 'prompt_embedding':7, 'suggestion_text_features':8, 'prompt_text_features':9,  'statename':10}
    df_observations_features, df_observations_labels = get_features_labels(path, features_to_keep, label_index, REMOVE_S_AND_R)


    # split into train and test
    SEQUENCE_MODE = False # keep session as a sequence or split it into events
    SPLIT_BY_USER = bool(splitbyusers) # otherwise split by session uniformly
    ADD_PREVIOUS_STATES = True
    PREDICT_ACTION = True # Otherwise predict time in state
    NORMALIZE_DATA = False # normalize data
    test_percentage = args.testpercentage
    val_percentage = args.valpercentage
    previous_states_to_keep = 3
    if not PREDICT_ACTION and SPLIT_BY_USER:
        raise ValueError('Cannot predict time and split by user')



    X_train, X_test, X_val, y_train, y_test, y_val = process_data(df_observations_features, df_observations_labels,
    REMOVE_S_AND_R, SEQUENCE_MODE, SPLIT_BY_USER, ADD_PREVIOUS_STATES, PREDICT_ACTION, NORMALIZE_DATA,
    test_percentage, val_percentage, previous_states_to_keep) 



    # train model
    if PREDICT_ACTION:
        if use_gpu:
            model = XGBClassifier(tree_method='gpu_hist')
        else:
            model = XGBClassifier()
        model.fit(X_train, y_train)
        # predict
        y_pred = model.predict(X_test)
        # evaluate
        print("Accuracy:", accuracy_score(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        confusion_matrix_act = confusion_matrix(y_test, y_pred)
        classification_report_act = classification_report(y_test, y_pred)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        y_pred_proba = model.predict_proba(X_test)
        y_pred_proba = y_pred_proba[:,1]
        print("AUC:", roc_auc_score(y_test, y_pred_proba))
        auc = roc_auc_score(y_test, y_pred_proba)
        pickle.dump([accuracy, confusion_matrix_act, classification_report_act, auc], open(output_path + '/action_prediction_results.pkl', 'wb'))
        model.save_model(output_path+ "/model_trained.json")




    # plot calibration curve
    if PREDICT_ACTION:

        y_pred_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr = calibration_curve(y_test, y_pred_proba, n_bins=10)
        # Plot perfectly calibrated
        plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
        
        # Plot model's calibration curve
        plt.plot(tpr, fpr, marker = '.', label = 'XGBoost')
        pickle.dump([tpr, fpr], open(output_path + '/xgb_calibration_curve.pkl', 'wb'))

        leg = plt.legend(loc = 'upper left')
        plt.xlabel('Average Predicted Probability in each bin')
        plt.ylabel('Ratio of positives')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(output_path + '/calibration_curve.pdf',dpi=1000)    
        plt.clf()


    if PREDICT_ACTION:
        # print a curve where x axis is ration and y axis is accuracy
        coverages = []
        accuracies = []
        aucs = []
        for treshold in np.arange(0.01, 1, 0.01):
            treshold_high = treshold
            y_pred_proba = model.predict_proba(X_test)[:,1]
            y_pred_proba = np.array([max(y_pred_proba[i], 1- y_pred_proba[i]) for i in range(len(y_pred_proba))])
        
            y_pred = model.predict(X_test)
            y_pred_high_confidence = y_pred[y_pred_proba > treshold_high]
            y_pred_proba_high_confidence = y_pred_proba[y_pred_proba > treshold_high]
            y_test_high_confidence = y_test[y_pred_proba > treshold_high]
            coverages.append(len(y_pred_high_confidence)/len(y_pred))
            accuracies.append(accuracy_score(y_test_high_confidence, y_pred_high_confidence))
        # pickle data
        pickle.dump([coverages, accuracies], open(output_path + '/xgb_coverage_accuracy.pkl', 'wb'))
        plt.plot(coverages, accuracies)
        plt.xlabel('Coverage (based on tresholding model confidence)')
        plt.ylabel('Accuracy')
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # more detailed y axis
        plt.savefig(output_path + '/acc_vs_coverage.pdf',dpi=1000)  
        plt.clf()

    # learning curve
    if PREDICT_ACTION:
        # empty 2 d array
        training_izes = []
        max_trials = 1
        training_data_percentage = np.array([0.005,0.01,0.05,0.1,0.25,0.5,0.75,0.99])
        accuracies  = [[] for _ in range(max_trials)]
        aucs = [[] for _ in range(max_trials)]
        for trial in range(max_trials):
            training_sizes = []
            for split_percentage in training_data_percentage:
                # split train data using sklearn
                _, X_train_frac, _, y_train_frac = train_test_split(X_train, y_train, test_size=split_percentage)
                # train model
                if use_gpu:
                    model = XGBClassifier(tree_method='gpu_hist')
                else:
                    model = XGBClassifier()
                model.fit(X_train_frac, y_train_frac)
                # predict
                y_pred = model.predict(X_test)
                # evaluate
                accuracies[trial].append(accuracy_score(y_test, y_pred))
                aucs[trial].append(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
                training_sizes.append(len(X_train_frac))
        
        # plot with error bars and means
        accuracies = np.array(accuracies)
        aucs = np.array(aucs)
        pickle.dump([training_data_percentage, accuracies], open(output_path + '/xgb_learning_curve.pkl', 'wb'))
        plt.errorbar(training_data_percentage, [np.mean(accuracies[:,i]) for i in range(len(accuracies[0]))], yerr=[np.std(accuracies[:,i]) for i in range(len(accuracies[0]))])
        plt.xlabel('Training Data Size')
        plt.ylabel('Accuracy')
        plt.plot(0, np.mean(y_test), 'o', label='base rate')
        plt.legend()
        plt.savefig(output_path + '/learning_curve.pdf',dpi=1000)  
        plt.clf()

   
main()