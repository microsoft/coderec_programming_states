import math
import numpy as np
from tqdm import tqdm

def entropy_single_probability(prob):
    if prob <= 0 or prob >= 1:
        raise ValueError("Probability must be between 0 and 1 (exclusive).")
    entropy = - (prob * math.log2(prob) + (1 - prob) * math.log2(1 - prob))
    return entropy

class CDHF:
    def __init__(self, cond_acc_constraint = 0.95, lambdaa = 0.5, step_size_entropy= 0.01, set_size_m = 100, m1_use_constraint = None):
        self.threshold_r = 0.1
        self.threshold_m1 = 0.5
        self.threshold_m2 = 0.5
        self.cond_acc_constraint = cond_acc_constraint
        self.lambdaa = lambdaa
        self.step_size_entropy = step_size_entropy
        self.set_size_m = set_size_m
        self.m1_use_constraint = m1_use_constraint
        self.loose_ness = 0.05
        print(f'CDHF initialized with cond_acc_constraint = {cond_acc_constraint} and lambda = {lambdaa}')


    def predict(self, yprob1, yprob2):
        '''
        yprob1: probability of model 1
        yprob2: probability of model 2
        return: ypred, yprob, r_pred
        '''
        entropy_y1 = entropy_single_probability(yprob1)
        ypred1 = (yprob1 < self.threshold_m1) *1.0
        ypred2 = (yprob2 < self.threshold_m2) *1.0
        r_pred = 1
        if entropy_y1 < self.threshold_r:
            r_pred = 0
        if r_pred:
            return ypred2, yprob2, r_pred
        else:
            return ypred1, yprob1, r_pred
    
    def predict_batch(self, yprobs1, yprobs2):
        # same as predict but for batch
        ypreds = np.array([])
        yprobs = np.array([])
        r_preds = np.array([])
        for yprob1, yprob2 in zip(yprobs1, yprobs2):
            ypred, yprob, r_pred = self.predict(yprob1, yprob2)
            ypreds = np.append(ypreds, ypred)
            yprobs = np.append(yprobs, yprob)
            r_preds = np.append(r_preds, r_pred)
        return ypreds, yprobs, r_preds



    def loss_function(self, yprobs1, yprobs2, labels):
        ypreds, yprobs, r_preds = self.predict_batch(yprobs1, yprobs2)
        loss = 0
        loss = np.mean(r_preds) * self.lambdaa + (1-self.lambdaa) * (1-np.mean(ypreds ))
        # check if false negative rate constraint is satisfied
        cond_acc = np.mean(ypreds[ypreds == 1] == labels[ypreds == 1])
        if cond_acc < self.cond_acc_constraint:
            return 9999
        return loss


    def get_metrics(self, yprobs1, yprobs2, labels):
        ypreds, yprobs, r_preds = self.predict_batch(yprobs1, yprobs2)
        # get accuracy
        accuracy = np.mean(ypreds == labels)
        # get coverage
        coverage = np.mean(r_preds)
        # get false negative rate
        fnr = np.mean((ypreds == 0) & (labels == 1))
        # get false positive rate
        fpr = np.mean((ypreds == 1) & (labels == 0))
        # when preds=1 the accuracy
        cond_acc = np.mean(ypreds[ypreds == 1] == labels[ypreds == 1])
        # how often we pred 1
        pred_1 = np.mean(ypreds)
        metrics_dict = {'accuracy overall': accuracy, "often hidden":pred_1, 'cond accuracy': cond_acc, 'coverage with m1': 1-coverage, 'fnr': fnr, 'fpr': fpr}
        return metrics_dict

    def model_accuracies(self, yprobs1, yprobs2, labels):
        ypreds1 = (yprobs1 < 0.5) *1.0
        ypreds2 = (yprobs2 < 0.5) *1.0
        accuracy1 = np.mean(ypreds1 == labels)
        accuracy2 = np.mean(ypreds2 == labels)
        print(f'Accuracy of model 1: {accuracy1}, Accuracy of model 2: {accuracy2} ')



    def find_threshold_for_conditional_accuracy(self, probs, labels, target_accuracy=0.95):
        # probs should be probability for it being a 1: rejected
        zipped_list = sorted(list(zip(probs,labels)), key = lambda x: x[0])
        # get sorted probabilities and labels
        sorted_probs = np.array([x[0] for x in zipped_list])
        sorted_labels = np.array([x[1] for x in zipped_list])
        num_samples = len(sorted_labels)
        best_threshold = 0 
        acc_so_far = 0
        # Iterate through the sorted probabilities and labels to find the threshold
        for i in range(num_samples):
            acc_so_far += sorted_labels[i]
            if acc_so_far / (i + 1) >= target_accuracy:
                best_threshold = sorted_probs[i]
                
        return best_threshold


    def fit(self,  model1_probs, model2_probs, labels):
        step_size = self.step_size_entropy
        best_loss = 9999
        loose_ness = self.loose_ness
        entropies_1 = np.array([entropy_single_probability(prob) for prob in model1_probs])
        lb_model1 = self.find_threshold_for_conditional_accuracy(model1_probs, labels,  target_accuracy=self.cond_acc_constraint+loose_ness)
        ub_model1 = self.find_threshold_for_conditional_accuracy(model1_probs, labels, target_accuracy=self.cond_acc_constraint-loose_ness)
        lb_model2 = self.find_threshold_for_conditional_accuracy(model2_probs, labels, target_accuracy=self.cond_acc_constraint+loose_ness)
        ub_model2 = self.find_threshold_for_conditional_accuracy(model2_probs, labels, target_accuracy=self.cond_acc_constraint-loose_ness)
        entropies_value_set = np.quantile(entropies_1, np.arange(0.00, 1.00001, step_size))
        # model1_probs_set is set of values from model1_probs that are between lb_model1 and ub_model1
        model1_probs_set = np.array([x for x in model1_probs if x >= lb_model1 and x <= ub_model1])
        model2_probs_set = np.array([x for x in model2_probs if x >= lb_model2 and x <= ub_model2])
        # pick only 1000 values from model1_probs_set and model2_probs_set
        print(f'entropies_value_set: {len(entropies_value_set)}, model1_probs_set: {len(model1_probs_set)}, model2_probs_set: {len(model2_probs_set)}')
        model1_probs_set = np.random.choice(model1_probs_set, self.set_size_m)
        model2_probs_set = np.random.choice(model2_probs_set, self.set_size_m)
        print(f'entropies_value_set: {len(entropies_value_set)}, model1_probs_set: {len(model1_probs_set)}, model2_probs_set: {len(model2_probs_set)}')
        best_threshold_r = 0
        best_threshold_m1 = 0
        best_threshold_m2 = 0

        if self.m1_use_constraint != None:
            entropy_constraint = np.quantile(entropies_1, self.m1_use_constraint)
            entropies_value_set = [entropy_constraint]

        for threshold_r in tqdm(entropies_value_set):
            r_preds = (entropies_1 > threshold_r) * 1.0
            mean_r_preds = np.mean(r_preds)
            for threshold_m1 in model1_probs_set:
                ypreds1 = (model1_probs < threshold_m1) * 1.0
                mean_ypreds1 = np.mean(ypreds1)
                for threshold_m2 in model2_probs_set:
                    ypreds2 = (model2_probs < threshold_m2) * 1.0
                    mean_ypreds2 = np.mean(ypreds2)
                    y_preds = r_preds * ypreds2 + (1-r_preds) * ypreds1
                    loss = mean_r_preds * self.lambdaa + (1- np.mean(y_preds ))*(1 - self.lambdaa)
                    if np.mean(y_preds) == 0:
                        cond_acc = 0
                    else:
                        cond_acc = np.mean(y_preds[y_preds == 1] == labels[y_preds == 1])
                    if cond_acc < self.cond_acc_constraint:
                        loss = 999
                    if loss < best_loss:
                        print(f'loss: {loss}, cond_acc: {cond_acc} r_preds mean: {mean_r_preds}, ypreds mean: {np.mean(y_preds)}')
                        best_loss = loss
                        best_threshold_r = threshold_r
                        best_threshold_m1 = threshold_m1
                        best_threshold_m2 = threshold_m2

        # Update the thresholds with the best values
        self.threshold_r = best_threshold_r
        self.threshold_m1 = best_threshold_m1
        self.threshold_m2 = best_threshold_m2
        print(self.get_metrics(model1_probs, model2_probs, labels))
        # Print the optimized thresholds
        print("Optimized threshold_r:", self.threshold_r)
        print("Optimized treshold_m1:", self.threshold_m1)
        print("Optimized treshold_m2:", self.threshold_m2)


