import numpy as np
import math

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes
        
        self.shadow_train_outputs, self.shadow_train_labels = shadow_train_performance
        self.shadow_test_outputs, self.shadow_test_labels = shadow_test_performance
        self.target_train_outputs, self.target_train_labels = target_train_performance
        self.target_test_outputs, self.target_test_labels = target_test_performance
        
        '''
        Determine corectness using the method described by Leino et al:
            - correct if predicted correctly
            - incorrect if predicted incorrectly
        Used for inference based on prediction correctness
        '''
        self.shadow_train_predictions = np.argmax(self.shadow_train_outputs, axis=1)
        self.shadow_test_predictions = np.argmax(self.shadow_test_outputs, axis=1)
        self.target_train_predictions = np.argmax(self.target_train_outputs, axis=1)
        self.target_test_predictions = np.argmax(self.target_test_outputs, axis=1)


        self.shadow_train_correctness = self._compute_correctness(self.shadow_train_predictions, self.shadow_train_labels)
        self.shadow_test_correctness = self._compute_correctness(self.shadow_test_predictions, self.shadow_test_labels)
        self.target_train_correctness = self._compute_correctness(self.target_train_predictions, self.target_train_labels)
        self.target_test_correctness = self._compute_correctness(self.target_test_predictions, self.target_test_labels)

        '''
        Separate the data needed for inference based on prediction confidence:
            each array will contain the predictions associated with each label
        '''
        self.shadow_train_confidence = self._get_label_predictions(self.shadow_train_labels, self.shadow_train_outputs)
        self.shadow_test_confidence = self._get_label_predictions(self.shadow_test_labels, self.shadow_test_outputs)
        self.target_train_confidence = self._get_label_predictions(self.target_train_labels, self.target_train_outputs)
        self.target_test_confidence = self._get_label_predictions(self.target_test_labels, self.target_test_outputs)

        # Compute the entropy using the normal method for each dataset
        
        self.shadow_train_entropy = self._compute_entropy(self.shadow_train_outputs)
        self.shadow_test_entropy = self._compute_entropy(self.shadow_test_outputs)
        self.target_train_entropy = self._compute_entropy(self.target_train_outputs)
        self.target_test_entropy = self._compute_entropy(self.target_test_outputs)
        
        # Compute the entropy using the proposed method for each dataset
        self.shadow_train_modified_entropy = self._compute_modified_entropy(self.shadow_train_outputs, self.shadow_train_labels)
        self.shadow_test_modified_entropy = self._compute_modified_entropy(self.shadow_test_outputs, self.shadow_test_labels)
        self.target_train_modified_entropy = self._compute_modified_entropy(self.target_train_outputs, self.target_train_labels)
        self.target_test_modified_entropy = self._compute_modified_entropy(self.target_test_outputs, self.target_test_labels)

        
    def _get_label_predictions(self, labels, predictions_array):
        aux = []
        for i in range(len(labels)):
            aux.append(predictions_array[i, labels[i]])
        result = np.array(aux)
        return result

    def _compute_correctness(self, predictions, labels):
        return (predictions==labels).astype(int)

    def _log_value(self, probs):
        logs = []
        for i in range(len(probs)):
            aux = []
            for j in range(len(probs[i])):
                aux.append(-math.log(probs[i][j]))
            logs.append(aux)
        return np.array(logs)
    
    def _compute_entropy(self, probs):
        aux = []
        for i in range(len(probs)):
            sum = 0
            for j in range(len(probs[i])):
                sum = sum + (-math.log(probs[i][j])) * probs[i][j]
            aux.append(sum)
        result = np.array(aux)
        return result
    
    def _compute_modified_entropy(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = probs
        for i in range(true_labels.size):
            for label in true_labels:
                modified_probs[i][label] = reverse_probs[i][label]
        modified_log_probs = log_reverse_probs
        for i in range(true_labels.size):
            for label in true_labels:
                modified_log_probs[i][label] = log_probs[i][label]
        result = []
        for i in range(len(modified_probs)):
            sum = 0
            for j in range(len(modified_probs[i])):
                sum = sum + (modified_probs[i][j] * modified_log_probs[i][j])
            result.append(sum)
        return np.array(result)
        
    def _count_values_greater_than(self, array, value):
        count = 0
        for i in range(len(array)):
            if array[i] >= value:
                count = count + 1
        return count
    
    def _count_values_less_than(self, array, value):
        count = 0
        for i in range(len(array)):
            if array[i] < value:
                count = count + 1
        return count

    def _compute_ratios(self, train_val, test_val, value):
        train_ratio = self._count_values_greater_than(train_val, value)/len(train_val)
        test_ratio = self._count_values_less_than(test_val, value)/len(test_val)
        return train_ratio, test_ratio
    
    def _threshold_setting(self, train_values, test_values):
        value_list = np.concatenate((train_values, test_values))
        threshold = 0
        max_acc = 0
        for value in value_list:
            train_ratio, test_ratio = self._compute_ratios(train_values, test_values, value)
            acc = 0.5*(train_ratio + test_ratio)
            if acc > max_acc:
                threshold = value
                max_acc = acc
        return threshold
    
    def _membership_inference_correctness(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        target_train_acc = np.sum(self.target_train_correctness)/len(self.target_train_correctness)
        target_test_acc = np.sum(self.target_test_correctness)/len(self.target_test_correctness)
        mem_inf_acc = 0.5*(target_train_acc + 1 - target_test_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=target_train_acc, acc3=target_test_acc) )
    
    def _membership_inference_feature(self, feature_name, shadow_train_values, shadow_test_values, target_train_values, target_test_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        target_train_memorised = 0
        target_test_non_memorised = 0
        for num in range(self.num_classes):
            thre = self._threshold_setting(shadow_train_values[self.shadow_train_labels==num], shadow_test_values[self.shadow_test_labels==num])
            target_train_memorised += self._count_values_greater_than(target_train_values[self.target_train_labels==num], thre)
            target_test_non_memorised += self._count_values_less_than(target_test_values[self.target_test_labels==num], thre)
        mem_inf_acc = 0.5*(target_train_memorised/len(self.target_train_labels) + target_test_non_memorised/len(self.target_test_labels))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=feature_name,acc=mem_inf_acc))
        return
    

    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._membership_inference_correctness()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._membership_inference_feature('confidence', self.shadow_train_confidence, self.shadow_test_confidence, self.target_train_confidence, self.target_test_confidence)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._membership_inference_feature('entropy', -self.shadow_train_entropy, -self.shadow_test_entropy, -self.target_train_entropy, -self.target_test_entropy)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._membership_inference_feature('modified entropy', -self.shadow_train_modified_entropy, -self.shadow_test_modified_entropy, -self.target_train_modified_entropy, -self.target_test_modified_entropy)