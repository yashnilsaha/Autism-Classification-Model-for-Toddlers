from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)  # Theta and x dot matrix multiplication
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient  # Calculating step size and new parameters
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)  # Calculation of loss function
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()



#read data
df=pd.read_csv("Autism.csv")

# remove unwanted columns
df.drop(['A1','A2','A3','A4','A7','A10','Age_Mons','Qchat-10-Score','Sex','Ethnicity','Jaundice','Family_mem_with_ASD','Case_No', 'Who completed the test'], axis = 1, inplace = True)
print(df.head)

#Preprocessing features to get them ready for modeling through encoding caterogical features

le = LabelEncoder()
columns = ['Class/ASD Traits ']
for col in columns:
    df[col] = le.fit_transform(df[col])

#print('after encode')

#print(df.columns)
X = df[["A6","A9","A5","A8"]]

X = np.array(X)
#print("X")
#print(X)
#print("length of X")
print(len(X))

y = df['Class/ASD Traits ']
print(y)
y = np.array(y)
print("y")

print("length of y")
print(len(y))
print(df.head)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.40, random_state=42)

modelAutism = LogisticRegression(lr=0.1, num_iter=1000)
modelAutism.fit(X_train, Y_train)

predsAutism = modelAutism.predict(X_test)
print("My model predictions ",predsAutism)
print("My model score ",(predsAutism == Y_test).mean())
print("My model theta ",modelAutism.theta)

from sklearn.linear_model import LogisticRegression

modelSklearn = LogisticRegression()
modelSklearn.fit(X_train, Y_train)
predsSklearn = modelSklearn.predict(X_test)
print("Sklearn predictions ",predsSklearn)
print("Sklearn model score ",(predsSklearn == Y_test).mean())




