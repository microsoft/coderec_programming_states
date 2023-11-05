import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from itertools import chain


def get_features_labels_user_study(path_df, features_to_keep, label_index, REMOVE_S_AND_R, predicting_time = False):
    df_observations, feature_dict= pickle.load(open(path_df, 'rb'))
    label_to_idx = {'Accepted': 0, 'Rejected': 1, 'Browsing': 2, 'Replay': 3, 'Shown': 4}
    idx_to_label = {0: 'Accepted', 1: 'Rejected', 2: 'Browsing', 3: 'Replay', 4: 'Shown'}
    # COMMENT THE LINE BELOW IF NOT USING USER STUDY DATA
    df_observations = [df_observations]
    df_observations_features = []
    df_observations_labels = []

    for i in tqdm(range(len(df_observations))):
        df_user = []
        df_user_labels = []
        for j in range(len(df_observations[i])):
            session_df = []
            session_labels = []
            for h in range(len(df_observations[i][j])):
                features_list = np.array(df_observations[i][j][h])[features_to_keep]
                # merge features
                features = []
                for f in features_list:
                    features.extend(f)
                features = np.array(features)
                if not predicting_time:
                    label = label_to_idx[np.array(df_observations[i][j][h])[label_index[0]]]
                else:
                    label = np.array(df_observations[i][j][h])[label_index[0]]
                if REMOVE_S_AND_R and label in [2,3,4]:
                    continue
                session_df.append(features)
                session_labels.append(label)
            if len(session_df) > 0:
                df_user.append(session_df)
                df_user_labels.append(session_labels)
        if len(df_user) > 0:
            df_observations_features.append(df_user)
            df_observations_labels.append(df_user_labels)
        # save space and delete df_observations[i]
        df_observations[i] = None
    df_observations_features =  df_observations_features[0]
    # make each element a list
    df_observations_features = [ [df_observations_features[i]] for i in range(len(df_observations_features))]
    df_observations_labels = df_observations_labels[0]
    # make each element a list
    df_observations_labels = [ [df_observations_labels[i]] for i in range(len(df_observations_labels))]
    return df_observations_features, df_observations_labels




def process_data_user_study(df_observations_features, df_observations_labels, REMOVE_S_AND_R, SEQUENCE_MODE,
 SPLIT_BY_USER, ADD_PREVIOUS_STATES, PREDICT_ACTION, NORMALIZE_DATA,
  test_percentage, val_percentage, previous_states_to_keep, random_state =66):

    def process_session_list(df_features_list, df_labels_list, previous_states_to_keep):
        df_features_subset = []
        df_labels_subset = []
        for i in range(len(df_features_list)):
            for k in range(len(df_features_list[i])):
                features = []
                labels = []
                for j in range(len(df_features_list[i][k])):
                        features.append(df_features_list[i][k][j])
                        labels.append(df_labels_list[i][k][j])
                if len(features) > 0:
                    df_features_subset.append(np.array(features))
                    df_labels_subset.append(np.array(labels))

        df_features_subset = np.array(df_features_subset)
        df_labels_subset = np.array(df_labels_subset)
        df_features_subset_append_prev = []
        df_labels_subset_append_prev = []

        if ADD_PREVIOUS_STATES:
            for k in range(len(df_features_subset)):
                features = []
                labels = []
                for j in range(len(df_features_subset[k])):
                    feature_construction= df_features_subset[k][j]
                    # get previous 5 labels
                    previous_labels = np.ones(previous_states_to_keep)
                    previous_measurements = np.zeros(8*previous_states_to_keep)
                    idx = 0
                    for l in range(j-previous_states_to_keep,j):
                        if l >= 0:
                            previous_labels[idx] = df_labels_subset[k][l]
                            previous_measurements[idx*8:(idx+1)*8] = df_features_subset[k][l][:8]
                        idx += 1
                    feature_construction = np.concatenate((df_features_subset[k][j], previous_labels, previous_measurements))

                    features.append(feature_construction)
                    labels.append(df_labels_subset[k][j])
                if len(features) > 0:
                    df_features_subset_append_prev.append(np.array(features))
                    df_labels_subset_append_prev.append(np.array(labels))
            return df_features_subset_append_prev, df_labels_subset_append_prev

        return df_features_subset, df_labels_subset


    if not SPLIT_BY_USER:
        df_features_subset, df_labels_subset = process_session_list(df_observations_features, df_observations_labels, previous_states_to_keep)
        X_train, X_test, y_train, y_test = train_test_split(df_features_subset, df_labels_subset, test_size=test_percentage, random_state=random_state)
        # split into validation and test
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percentage/(1-test_percentage), random_state=random_state)
        # unflatten X_train and X_test and y_train and y_test
        if not SEQUENCE_MODE:
            X_train = np.array(list(chain.from_iterable(X_train)))
            X_test = np.array(list(chain.from_iterable(X_test)))
            y_train = np.array(list(chain.from_iterable(y_train)))
            y_test = np.array(list(chain.from_iterable(y_test)))
            X_val = np.array(list(chain.from_iterable(X_val)))
            y_val = np.array(list(chain.from_iterable(y_val)))
            if NORMALIZE_DATA:
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)
    else:
        # split on users first
        X_train, X_test, y_train, y_test = train_test_split(df_observations_features, df_observations_labels, test_size=test_percentage, random_state=random_state)
        # split into validation and test
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percentage/(1-test_percentage), random_state=random_state)
        # print lenght of train test and val
        X_train, y_train = process_session_list(X_train, y_train, previous_states_to_keep)
        X_test, y_test = process_session_list(X_test, y_test,  previous_states_to_keep)
        X_val, y_val = process_session_list(X_val, y_val, previous_states_to_keep)

        if not SEQUENCE_MODE:
            X_train = np.array(list(chain.from_iterable(X_train)))
            X_test = np.array(list(chain.from_iterable(X_test)))
            y_train = np.array(list(chain.from_iterable(y_train)))
            y_test = np.array(list(chain.from_iterable(y_test)))
            X_val = np.array(list(chain.from_iterable(X_val)))
            y_val = np.array(list(chain.from_iterable(y_val)))
            if NORMALIZE_DATA:
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)
    # print lenght of train test and val
    print('X_train: ', len(X_train))
    print('X_test: ', len(X_test))
    print('X_val: ', len(X_val))
    return X_train, X_test, X_val, y_train, y_test, y_val



def get_features_labels(path_df, features_to_keep, label_index, REMOVE_S_AND_R, predicting_time = False):
    df_observations, feature_dict= pickle.load(open(path_df, 'rb'))
    label_to_idx = {'Accepted': 0, 'Rejected': 1, 'Browsing': 2, 'Replay': 3, 'Shown': 4}
    idx_to_label = {0: 'Accepted', 1: 'Rejected', 2: 'Browsing', 3: 'Replay', 4: 'Shown'}

    df_observations_features = []
    df_observations_labels = []

    for i in tqdm(range(len(df_observations))):
        df_user = []
        df_user_labels = []
        for j in range(len(df_observations[i])):
            session_df = []
            session_labels = []
            for h in range(len(df_observations[i][j])):
                features_list = np.array(df_observations[i][j][h])[features_to_keep]
                # merge features
                features = []
                for f in features_list:
                    features.extend(f)
                features = np.array(features)
                if not predicting_time:
                    label = label_to_idx[np.array(df_observations[i][j][h])[label_index[0]]]
                else:
                    label = np.array(df_observations[i][j][h])[label_index[0]]
                if REMOVE_S_AND_R and label in [2,3,4]:
                    continue
                session_df.append(features)
                session_labels.append(label)
            if len(session_df) > 0:
                df_user.append(session_df)
                df_user_labels.append(session_labels)
        if len(df_user) > 0:
            df_observations_features.append(df_user)
            df_observations_labels.append(df_user_labels)
        # save space and delete df_observations[i]
        df_observations[i] = None

    return df_observations_features, df_observations_labels




def process_data(df_observations_features, df_observations_labels, REMOVE_S_AND_R, SEQUENCE_MODE,
 SPLIT_BY_USER, ADD_PREVIOUS_STATES, PREDICT_ACTION, NORMALIZE_DATA,
  test_percentage, val_percentage, previous_states_to_keep, random_state =66):

    def process_session_list(df_features_list, df_labels_list, previous_states_to_keep):
        df_features_subset = []
        df_labels_subset = []
        for i in range(len(df_features_list)):
            for k in range(len(df_features_list[i])):
                features = []
                labels = []
                for j in range(len(df_features_list[i][k])):
                        features.append(df_features_list[i][k][j])
                        labels.append(df_labels_list[i][k][j])
                if len(features) > 0:
                    df_features_subset.append(np.array(features))
                    df_labels_subset.append(np.array(labels))

        df_features_subset = np.array(df_features_subset)
        df_labels_subset = np.array(df_labels_subset)
        df_features_subset_append_prev = []
        df_labels_subset_append_prev = []

        if ADD_PREVIOUS_STATES:
            for k in range(len(df_features_subset)):
                features = []
                labels = []
                for j in range(len(df_features_subset[k])):
                    feature_construction= df_features_subset[k][j]
                    # get previous 5 labels
                    previous_labels = np.ones(previous_states_to_keep)
                    previous_measurements = np.zeros(8*previous_states_to_keep)
                    idx = 0
                    for l in range(j-previous_states_to_keep,j):
                        if l >= 0:
                            previous_labels[idx] = df_labels_subset[k][l]
                            previous_measurements[idx*8:(idx+1)*8] = df_features_subset[k][l][:8]
                        idx += 1
                    feature_construction = np.concatenate((df_features_subset[k][j], previous_labels, previous_measurements))

                    features.append(feature_construction)
                    labels.append(df_labels_subset[k][j])
                if len(features) > 0:
                    df_features_subset_append_prev.append(np.array(features))
                    df_labels_subset_append_prev.append(np.array(labels))
            return df_features_subset_append_prev, df_labels_subset_append_prev

        return df_features_subset, df_labels_subset


    if not SPLIT_BY_USER:
        df_features_subset, df_labels_subset = process_session_list(df_observations_features, df_observations_labels, previous_states_to_keep)
        X_train, X_test, y_train, y_test = train_test_split(df_features_subset, df_labels_subset, test_size=test_percentage, random_state=random_state)
        # split into validation and test
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percentage/(1-test_percentage), random_state=random_state)
        # unflatten X_train and X_test and y_train and y_test
        if not SEQUENCE_MODE:
            X_train = np.array(list(chain.from_iterable(X_train)))
            X_test = np.array(list(chain.from_iterable(X_test)))
            y_train = np.array(list(chain.from_iterable(y_train)))
            y_test = np.array(list(chain.from_iterable(y_test)))
            X_val = np.array(list(chain.from_iterable(X_val)))
            y_val = np.array(list(chain.from_iterable(y_val)))
            if NORMALIZE_DATA:
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)
    else:
        # split on users first
        X_train, X_test, y_train, y_test = train_test_split(df_observations_features, df_observations_labels, test_size=test_percentage, random_state=random_state)
        # split into validation and test
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_percentage/(1-test_percentage), random_state=random_state)

        X_train, y_train = process_session_list(X_train, y_train, previous_states_to_keep)
        X_test, y_test = process_session_list(X_test, y_test,  previous_states_to_keep)
        X_val, y_val = process_session_list(X_val, y_val, previous_states_to_keep)

        if not SEQUENCE_MODE:
            X_train = np.array(list(chain.from_iterable(X_train)))
            X_test = np.array(list(chain.from_iterable(X_test)))
            y_train = np.array(list(chain.from_iterable(y_train)))
            y_test = np.array(list(chain.from_iterable(y_test)))
            X_val = np.array(list(chain.from_iterable(X_val)))
            y_val = np.array(list(chain.from_iterable(y_val)))
            if NORMALIZE_DATA:
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                X_val = scaler.transform(X_val)
    # print lenght of train test and val
    print('X_train: ', len(X_train))
    print('X_test: ', len(X_test))
    print('X_val: ', len(X_val))
    return X_train, X_test, X_val, y_train, y_test, y_val