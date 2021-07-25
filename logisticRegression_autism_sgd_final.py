import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


df = pd.read_csv("Autism.csv")

# remove unwanted columns
df.drop(['A1','A2','A3','A4','A5','A7','A8','A10','Age_Mons','Qchat-10-Score','Sex','Ethnicity','Jaundice','Family_mem_with_ASD','Case_No', 'Who completed the test'], axis = 1, inplace = True)
print(df.head)

#Preprocessing features to get them ready for modeling through encoding caterogical features

le = LabelEncoder()
columns = ['Class/ASD Traits ']
for col in columns:
    df[col] = le.fit_transform(df[col])

X = df[["A9","A6"]]
X = np.array(X)
print("X", X)

print("length of X")
print(len(X))

Y = df['Class/ASD Traits ']
Y = np.array(Y)
print("Y", Y)
print("length of y")
print(len(Y))
dataset = []
dataset_train = []
dataset_test = []


target_label = 1 # choose the target label of autism
for index, x in enumerate(X):
    transform_label = None
    if Y[index] == target_label:
        transform_label = 1 # is the type
    else:
        transform_label = 0 # is not the type
    x = [x[0],x[1]]
    
    dataset.append((x,transform_label))
for i in range((int)(len(dataset)*0.6)):
    dataset_train.append(dataset[i])
for i in range((int)(len(dataset)*0.6),len(dataset)):
    dataset_test.append(dataset[i])

print("printing dataset")       
dataset = np.array(dataset)
print(dataset)
print(dataset_train)
print(dataset_test)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sgd(dataset, w):
    #run sgd randomly
    index = random.randint(0, len(dataset) - 1)
    x, y = dataset[index]
    x = np.array(x)
    error = sigmoid(w.T.dot(x))
    g = (error - y) * x
    return g

def cost(dataset, w):
    total_cost = 0
    for x,y in dataset:
        x = np.array(x)
        error = sigmoid(w.T.dot(x))
        total_cost += abs(y - error)
    return total_cost

def logistic_regression(dataset):
    w = np.zeros(2)  # Weights getting initialized
    limit = 1000 #number of iterations
    eta = 0.1 #learning rate
    costs = []
    for i in range(limit):
        current_cost = cost(dataset, w)  # gets cost adding up differences between y and error
        if i % 100 == 0:
            print ("Iteration = " + str(i/100 + 1) + ": current_cost = ", current_cost)
        costs.append(current_cost)
        w -= eta * sgd(dataset, w) # Calculating step size and new parameters
        eta = eta * 0.98 # decrease learning rate
    plt.plot(range(limit), costs)
    plt.show()
    return w,(limit, costs)

def main():
    #execute
    model_train = logistic_regression(dataset_train)

    #draw 
    ps = [v[0] for v in dataset_train]
    label = [v[1] for v in dataset_train]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #plot via label
    tpx=[]
    for index, label_value in enumerate(label):
        px=ps[index][0]
        py=ps[index][1]
        tpx.append(px)
        if label_value == 1:
            ax1.scatter(px, py, c='b', marker="o", label='O')
        else:
            ax1.scatter(px, py, c='r', marker="x", label='X')

    l = np.linspace(min(tpx),max(tpx))
    a,b = (-model_train[0][0]/model_train[0][1],model_train[0][0])
    ax1.plot(l, a*l + b, 'g-')
    #plt.legend(loc='upper left');
    plt.show()

    limit = model_train[1][0]
    costs = model_train[1][1]
    model_train = model_train[0]
    # calculate score
    probabilities_Y_test = []
    answer_Y_test = []
    predicted_Y_test = []
    for X,Y in dataset_test:
        answer_Y_test.append(Y)
        probabilities_Y_test.append(sigmoid(model_train.T.dot(X)))
    probabilities_Y_test = np.asarray(probabilities_Y_test)
    for i in range(len(probabilities_Y_test)):
        if probabilities_Y_test[i] > 0.5:
            predicted_Y_test.append(1)
        else:
            predicted_Y_test.append(0)
    print("Accuracy test: ", str(accuracy_score(answer_Y_test, predicted_Y_test) * 100)[:5], "%")
if __name__ == '__main__':
    main()