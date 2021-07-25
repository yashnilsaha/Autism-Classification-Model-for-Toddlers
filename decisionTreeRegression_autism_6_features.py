import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TreeNode:  # Tree_Node

    def __init__(self, data, output):
        self.data = data  # feature_name on which it split
        self.children = {}
        self.index = -1
        self.output = output  # majority class value

    def add_child(self, feature_val, obj):
        self.children[feature_val] = obj


class DecisionTreeClassifier:

    def __init__(self):
        self._root = None  # root of tree

    def __count__unique(self, Y):
        d = {}
        for i in Y:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1
        return d

    def __entropy(self, Y):

        freq = self.__count__unique(Y)
        tot = len(Y)
        sum = 0
        for i in freq:
            sum += ((-freq[i] / tot) * math.log2(freq[i] / tot))
        return sum

    def __gain_ratio(self, X, Y, selected_feature):

        orig_info = self.__entropy(Y)
        info_after_split = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        tot = df.shape[0]
        split_info = 0
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            info_after_split += (current_size / tot) * self.__entropy(df1[df1.shape[1] - 1])
            split_info += (-current_size / tot) * math.log2((current_size / tot))

        if split_info == 0:
            return math.inf
        info_gain = orig_info - info_after_split
        gain_ratio = info_gain / split_info

        return gain_ratio

    def __gini_(self, Y):

        freq = self.__count__unique(Y)
        tot = len(Y)
        sum = 1
        for i in freq:
            sum -= (freq[i] / tot) ** 2
        return sum

    def __gini_ratio(self, X, Y, selected_feature):

        orig_info = self.__gini_(Y)
        info_after_split = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        tot = df.shape[0]
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            info_after_split += (current_size / tot) * self.__gini_(df1[df1.shape[1] - 1])
        info_gain = orig_info - info_after_split
        return info_gain

    def __decisionTree(self, X, Y, features, level, metric, classes):

        if len(set(Y)) == 1:  # BASE condition 1
            print("level", level)
            output = None

            for i in classes:
                if i not in Y:
                    print("Count of", i, "=0")
                else:
                    output = i
                    print("Count of", i, len(Y))
            if metric == "gain_ratio":
                print("Current Entropy with gain_ratio", 0)
            elif metric == "gini_ratio":
                print("Current Entropy with gini_ratio", 0)
            return TreeNode(None, output)
        if len(features) == 0:  # BASE condition 2
            output = None
            freq = self.__count__unique(Y)
            max_count = -math.inf
            for i in classes:
                if i not in freq:
                    print("Count of", i, "=0")
                else:
                    if max_count < freq[i]:
                        max_count = freq[i]
                        output = i
            if metric == "gain_ratio":
                print("Current Entropy with gain_ratio", self.__entropy(Y))
            elif metric == "gini_ratio":
                print("Current Entropy with gini_ratio", self.__gini_(Y))
            return TreeNode(None, output)
        max_gain = -math.inf
        final_feature = None

        for i in features:  # check which feature Give maximum Gain
            if metric == "gain_ratio":
                current_gain = self.__gain_ratio(X, Y, i)
            if metric == "gini_ratio":
                current_gain = self.__gini_ratio(X, Y, i)
            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = i
        freq = self.__count__unique(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i not in freq:
                print("Count of", i, "=0")
            else:
                if freq[i] > max_count:
                    output = i
                    max_count = freq[i]
        if metric == "gain_ratio":
            # print("Current Entropy is =",self.__entropy(Y))
            # print("Splitting on feature  X[",final_feature,"] with gain ratio ",max_gain,sep="")
            print()
        elif metric == "gini_index":
            # print("Current Gini Index is =",self.__gini_index(Y))
            # print("Splitting on feature  X[",final_feature,"] with gini gain ",max_gain,sep="")
            print()
        current_node = TreeNode(final_feature, output)
        unique_val = set(X[:, final_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        idx = features.index(final_feature)
        features.remove(final_feature)

        for i in unique_val:  # make children
            df1 = df[df[final_feature] == i]
            node = self.__decisionTree(df1.iloc[:, 0:df1.shape[1] - 1].values, df1.iloc[:, df1.shape[1] - 1].values,
                                       features, level + 1, metric, classes)
            current_node.add_child(i, node)
        features.insert(idx, final_feature)

        return current_node

    def fit(self, X, Y, metric="gain_ratio"):

        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0

        self._root = self.__decisionTree(X, Y, features, level, metric, classes)
        return self._root

    def predict_for(self, data, node):

        if len(node.children) == 0:
            return node.output
        val = data[node.data]
        if val not in node.children:
            return node.output

        return self.predict_for(data, node.children[val])

    def predict(self, X):

        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.predict_for(X[i], self._root)
        return Y

    def score(self, X, Y):
        Y_pred = self.predict(X)
        cont = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                cont += 1
        return cont / len(Y_pred)


#read data
df=pd.read_csv("Autism.csv")

# remove unwanted columns
df.drop(['A2','A4','A7','A10','Age_Mons','Qchat-10-Score','Sex','Ethnicity','Jaundice','Family_mem_with_ASD','Case_No', 'Who completed the test'], axis = 1, inplace = True)

# Preprocessing features to get them ready for modeling through encoding categorical features

le = LabelEncoder()
columns = ['Class/ASD Traits ']
for col in columns:
    df[col] = le.fit_transform(df[col])

X_aut = df[["A9", "A5", "A6","A8","A1","A3"]]
X_aut = np.array(X_aut)

Y_aut = df['Class/ASD Traits ']
Y_aut=np.array(Y_aut)

X_train, X_test, Y_train, Y_test = train_test_split(X_aut, Y_aut, test_size=0.40, random_state=42)
start_time = time.time()
clsAutism = DecisionTreeClassifier()

decisionTreeModel = clsAutism.fit(X_train,Y_train)
print("prediction for autism")
prediction = clsAutism.predict(X_test)
print('score for autism ')
print(clsAutism.score(X_test, Y_test))
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
print("Mean squared error = ", mean_squared_error(Y_test, prediction))

from sklearn.tree import DecisionTreeClassifier

start_time_sklearn = time.time()
clsSklearn = DecisionTreeClassifier()
sklearnModel = clsSklearn.fit(X_train,Y_train)
print("prediction for autism from sklearn")
print(clsSklearn.predict(X_test))
print('score for autism from sklearn')
print(clsSklearn.score(X_test,Y_test))
end_time_sklearn = time.time()
print("--- %s seconds ---" % (end_time_sklearn - start_time_sklearn))

