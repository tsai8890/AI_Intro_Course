from argparse import ArgumentParser
from typing import Tuple, Union, List, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

def data_preprocessing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return the preprocessed data (X_train, X_test, y_train, y_test). 
    You will need to remove the "Email No." column since it is not a valid feature.
    """
    data = data.iloc[:, 1:]
    nrows, ncols = data.shape

    train_x = data.iloc[:int(nrows*0.8), :ncols-1]
    train_y = data.iloc[:int(nrows*0.8), -1]
    test_x = data.iloc[int(nrows*0.8):, :ncols-1]
    test_y = data.iloc[int(nrows*0.8):, -1]
    return train_x, test_x, train_y, test_y    

class TreeNode:
    def __init__(self, field:str = None, threshold:int = None, 
                 left = None, right = None, leaf_label:int = None):
        self.field = field
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_label = leaf_label

class DecisionTree:
    def __init__(self):
        self.rootNode = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        "Fit the model with training data"
        self.rootNode = self.fit_helper(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        "Make predictions for the testing data"

        if self.rootNode == None:
            print('The model had not been trained!!!')
            return None

        if len(X.shape) < 2:
            print('The data ready for prediction should be in DataFrame[B, D1, D2, ...]')
            print('Where B stands for batch size, even if only one data needed, you should set B = 1')
            return None

        pred_Y = []
        nDatas = X.shape[0]
        for i in range(nDatas):
            x = X.iloc[i]
            pred_Y.append(self.predict_helper(x, self.rootNode))
        return pd.Series(pred_Y)

    def predict_helper(self, x: pd.DataFrame, curNode: TreeNode) -> Any:
        if curNode.leaf_label != None:
            return curNode.leaf_label
        else:
            filter = curNode.field
            threshold = curNode.threshold
            if x.loc[filter] <= threshold:
                return self.predict_helper(x, curNode.left)
            else:
                return self.predict_helper(x, curNode.right)

    def fit_helper(self, X: pd.DataFrame, y: pd.DataFrame) -> TreeNode:
        nrows, ncols = X.shape
        total_positive = X.loc[y == 1].shape[0]
        total_negative = nrows - total_positive

        if total_positive == 0 or total_negative == 0:
            label = 1 if total_positive != 0 else -1
            return TreeNode(leaf_label=label)

        best_col = -1
        best_threshold = -1
        best_confusion = -1

        for i in range(ncols):
            data = pd.concat([X.iloc[:, i], y], axis=1)
            data = data.sort_values(by=[data.columns[0]])
            thresholds = self.createThresholds(data, 0)
            if len(thresholds) == 0:
                continue
            
            accum_positive, accum_negative = self.getAccumStat(data, thresholds, 1)
            cur_best_threshold = -1
            cur_best_confusion = -1

            len_ = len(thresholds)
            for j in range(len_):
                left_positive, left_negative = accum_positive[j], accum_negative[j]
                right_positive, right_negative = total_positive - left_positive, total_negative - left_negative
                
                confusion_value = self.totalConfusion(left_positive, left_negative, right_positive, right_negative)
                if cur_best_confusion == -1 or confusion_value < cur_best_confusion:
                    cur_best_confusion = confusion_value
                    cur_best_threshold = thresholds[j]

            if best_confusion == -1 or cur_best_confusion < best_confusion:
                best_confusion = cur_best_confusion
                best_threshold = cur_best_threshold
                best_col = data.columns[0]

        left_subtree_X = X[X.loc[:, best_col] <= best_threshold]
        right_subtree_X = X[X.loc[:, best_col] > best_threshold]
        # left_subtree_X = left_subtree_X.drop([best_col], axis=1)
        # right_subtree_X = right_subtree_X.drop([best_col], axis=1)
        left_subtree_y = y[X.loc[:, best_col] <= best_threshold]
        right_subtree_y = y[X.loc[:, best_col] > best_threshold]
        
        # print(f'Dividing Point: {best_col}')
        # print(f'Threshold: {best_threshold}')
        # print(f'left: {left_subtree_X.shape[0]}, right: {right_subtree_X.shape[0]}')
        # print()

        head = TreeNode(best_col, best_threshold, None, None, None)
        head.left = self.fit_helper(left_subtree_X, left_subtree_y)
        head.right = self.fit_helper(right_subtree_X, right_subtree_y)
        return head

    def createThresholds(self, data: pd.DataFrame, target_col: int) -> List[int]:
        data = data.drop_duplicates(subset=[data.columns[0]])
        thresholds = []
        for i in range(data.shape[0]-1):
            thresholds.append((data.iloc[i][target_col] + data.iloc[i+1][target_col]) / 2)
        return thresholds

    def getAccumStat(self, data: pd.DataFrame, thresholds: List[int], label_col: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        nrows = data.shape[0]
        accum_positive = []
        accum_negative = []

        curPos = 0
        cur_positive = 0
        for i in range(len(thresholds)):
            while curPos < nrows and data.iloc[curPos][0] <= thresholds[i]:
                if data.iloc[curPos][label_col] == 1:
                    cur_positive += 1
                curPos += 1 
            
            accum_positive.append(cur_positive)
            accum_negative.append(curPos - cur_positive)
        return accum_positive, accum_negative

    def confusion(self, positive: int, negative: int) -> float:
        total = positive + negative
        return 1 - (positive / total)**2 - (negative / total)**2

    def totalConfusion(self, left_positive: int, left_negative: int, right_positive: int, right_negative: int) -> float:
        left_total = left_positive + left_negative
        right_total = right_positive + right_negative
        total = left_total + right_total

        return (left_total / total) * self.confusion(left_positive, left_negative) \
             + (right_total / total) * self.confusion(right_positive, right_negative)

class RandomForest:
    "Add more of your code here if you want to"
    def __init__(self, seed: int = 42, num_trees: int = 5):
        self.num_trees = num_trees
        self.trees = []
        self.classes = [-1, 1]
        np.random.seed(seed)

    def bagging(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "DO NOT modify this function. This function is deliberately given to make your result reproducible."
        index = np.random.randint(0, X.shape[0], int(X.shape[0] / 2))
        return X.iloc[index, :], y.iloc[index]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.trees = []
        for i in range(self.num_trees):
            train_X, train_y = self.bagging(X, y)
            newTree = DecisionTree()
            newTree.fit(train_X, train_y)
            self.trees.append(newTree)

    def predict(self, X) -> Any:
        all_results = []
        for i in range(self.num_trees):
            pred_Y = self.trees[i].predict(X)
            all_results.append(pred_Y.tolist())

        nDatas = X.shape[0]
        ensemble_results = []
        for i in range(nDatas):
            stat = dict()
            for j in range(self.num_trees):
                if all_results[j][i] not in stat:
                    stat[all_results[j][i]] = 1
                else:
                    stat[all_results[j][i]] += 1

            best_choice = None
            for c in stat:
                if best_choice == None or stat[c] > stat[best_choice]:
                    best_choice = c
            ensemble_results.append(best_choice)

        return pd.Series(ensemble_results)

def accuracy_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the accuracy score
    """
    y_pred = pd.Series(y_pred)
    y_label = pd.Series(y_label)
    if y_pred.shape[0] != y_label.shape[0]:
        print('accuracy_score: Two arrays have different shapes!!')
        return None
    
    y_pred.index = list(range(y_pred.shape[0]))
    y_label.index = list(range(y_label.shape[0]))
    return sum(y_pred == y_label) / len(y_pred)

def f1_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the F1 score
    """
    y_pred = pd.Series(y_pred)
    y_label = pd.Series(y_label)
    if y_pred.shape[0] != y_label.shape[0]:
        print('f1_score: Two arrays have different shapes!!')
        return None
    
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(y_pred.shape[0]):
        if y_pred.iloc[i] == y_label.iloc[i]:
            if y_pred.iloc[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y_pred.iloc[i] == 1:
                FP += 1
            else:
                FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def cross_validation(model: Union[LogisticRegression, DecisionTree, RandomForest], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> Tuple[float, float]:
    """
    Test the generalizability of the model with 5-fold cross validation
    Return the mean accuracy and F1 score
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=folds)
    f1, acc = 0, 0
    
    for train_index, val_index in kf.split(X, y):
        X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
        X_val, y_val = X.iloc[val_index, :], y.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # print(f'accuracy_score = {accuracy_score(y_pred, y_val)}')
        # print(f'f1_score = {f1_score(y_pred, y_val)}')
        acc += accuracy_score(y_val, y_pred)
        f1 += f1_score(y_val, y_pred)

    return acc / folds, f1 / folds

def cross_validation_byhand(model: Union[LogisticRegression, DecisionTree, RandomForest], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> Tuple[float, float]:
    nDatas = X.shape[0]
    f1, acc = 0, 0

    for fold_no in range(folds):
        X_train = X.iloc[[i for i in range(nDatas) if i % folds != fold_no]]
        y_train = y.iloc[[i for i in range(nDatas) if i % folds != fold_no]]
        X_val = X.iloc[[i for i in range(nDatas) if i % folds == fold_no]]
        y_val = y.iloc[[i for i in range(nDatas) if i % folds == fold_no]]

        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_val))

        # print(f'accuracy_score = {accuracy_score(y_pred, y_val)}')
        # print(f'f1_score = {f1_score(y_pred, y_val)}')
        f1 += f1_score(y_pred, y_val)
        acc += accuracy_score(y_pred, y_val)
    
    return acc / folds, f1 / folds

def tune_random_forest(choices: List[int], X: pd.DataFrame, y: pd.DataFrame) -> int:
    """
    choices: List of candidates for the number of decision trees in the random forest
    Return the best choice
    """
    best_f1 = -1
    best_choice = -1

    for choice in choices:
        random_forest = RandomForest(num_trees=choice)
        acc, f1 = cross_validation(random_forest, X, y, folds=5)
        print(f'num_trees={choice}: accuracy={acc}, f1_score={f1}')
        
        if best_f1 == -1 or f1 > best_f1:
            best_f1 = f1
            best_choice = choice  
    
    return best_choice

def main(args):
    """
    This function is provided as a head start
    TA will use his own main function at test time.
    """

    """ Data Preprocessing """
    import sys
    sys.stdout = open('./test.out', 'w')
    sys.stderr = sys.stdout

    data = pd.read_csv(args.data_path)
    print(data.head())
    print(data['Prediction'].value_counts())
    print()
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    
    logistic_regression = LogisticRegression(solver='liblinear', max_iter=500)
    decision_tree = DecisionTree()
    random_forest = RandomForest(num_trees=5)

    models = [logistic_regression, decision_tree, random_forest]
    model_names = ['logistic_regression', 'decision_tree', 'random_forest']
    best_f1, best_model, best_model_name = -1, None, None

    """ Cross Validation """
    print('-*-*- Cross Validation between logistics_regression, decision_tree, random_forest -*-*-')
    for i, model in enumerate(models):
        accuracy, f1 = cross_validation(model, X_train, y_train, 5)
        print(f'{model_names[i]}')
        print(f'accuracy: {accuracy}, f1_score: {f1}')
        print()

        if f1 > best_f1:
            best_f1, best_model, best_model_name = f1, model, model_names[i]
    
    print(f'-*-*- best model in five fold cross validation: {best_model_name} -*-*-')
    print()

    """ Use best model to predict X_test """
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(f'-*-*- {best_model_name} -*-*-')
    print(f'accuracy: {accuracy_score(y_pred, y_test)}, f1_score: {f1_score(y_pred, y_test)}')
    print()

    """ Tune the number of trees in the random forest"""
    print('-*-*- Tuning Random Forest -*-*-')
    best_num_trees = tune_random_forest([5, 11, 17], X_train, y_train)
    print(f'Best number of trees in the choices of [5, 11, 17]: {best_num_trees}')
    print()
    sys.stdout.close()

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./emails.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments())