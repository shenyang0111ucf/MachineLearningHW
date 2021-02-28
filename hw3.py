import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection._validation import cross_val_score
from sklearn.svm import SVC
#Q2################################################################################
X = np.array([[-1,1],[-1,-1],[1,-1],[1,1]])
y = [-1,1,1,-1]
  
SVCClf = SVC(kernel = 'linear', C=1000, gamma = 'scale', shrinking = False)
SVCClf.fit(X, y)
 
w = SVCClf.coef_[0]
print(w)
 
k = -w[0] / w[1]
 
xx = np.linspace(-2,2)
yy = k * xx - SVCClf.intercept_[0] / w[1]
 
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
 
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim([-2, 2])
plt.show()
#Q5##################################################################################
X = np.array([[1,1],[2,2],[2,0],[0,0],[1,0],[0,1]])
y = [1,1,1,-1,-1,-1]
  
SVCClf = SVC(kernel = 'linear', C=1000)
SVCClf.fit(X, y)
 
w = SVCClf.coef_[0]
print(w)
 
k = -w[0] / w[1]
 
xx = np.linspace(-1,3)
yy = k * xx - SVCClf.intercept_[0] / w[1]
 
h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
 
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim([-1,3])
plt.ylim([-1,3])
plt.show()
#Q7###################################################################################
#Use your own folder when testing
train_df = pd.read_csv('D:/HWData/train.csv')
test_df = pd.read_csv('D:/HWData/test.csv')

#These codes from HW2 Task1 Q1 and Q2 After feature selection ####################################################################################
fareTrainData = list(filter(lambda x: not pd.isnull(x), train_df['Fare']))
topFare = list(pd.Series(fareTrainData).mode())[0]
#print('mode for Fare:', topFare)
updatedFareData = train_df['Fare'].apply(lambda x: topFare if pd.isnull(x) else x)
train_df.update(updatedFareData)
updatedFareData = train_df['Fare'].apply(lambda x: 0 if x>-0.001 and x<=7.91 else (1 if x>7.91 and x<=14.454 else (2 if x>14.454 and x<=31 else 3)))
train_df.update(updatedFareData)

ageTrainData = list(filter(lambda x: not pd.isnull(x), train_df['Age']))
avgAge = sum(ageTrainData)/len(ageTrainData)

updatedAgeData = train_df['Age'].apply(lambda x: avgAge if pd.isnull(x) else x)
train_df.update(updatedAgeData)

updatedAgeData = train_df['Age'].apply(lambda x: 0 if x>=0 and x<=4 else (1 if x>4 and x<=15 else (2 if x>15 and x<=25 else (3 if x>25 and x<=40 else (4 if x>40 and x<=65 else 5)))))
train_df.update(updatedAgeData)

sexData = list(filter(lambda x: not pd.isnull(x), train_df[train_df.columns[4]]))
sexNumData = train_df[train_df.columns[4]].apply(lambda x: 1 if x == 'female' else 0)#female=1 male=0
#print('Sex correlation',pd.Series(sexNumData).corr(train_df[train_df.columns[1]]))
train_df.update(sexNumData)

updatedEmbarkedData = train_df['Embarked'].apply(lambda x: 'S' if pd.isnull(x) else x)# use mode to fill null value
train_df.update(updatedEmbarkedData)
updatedEmbarkedData = train_df['Embarked'].apply(lambda x: int(1) if x == 'S' else (int(2) if x == 'Q' else int(3)) )#S=1 Q=2 C=3
train_df.update(updatedEmbarkedData)
#print('Embarked correlation',(train_df['Embarked'].astype('int32')).corr(train_df[train_df.columns[1]]))

corr_matrix = train_df.corr(method='pearson')
#print(corr_matrix['Survived']['Pclass'])
#print(corr_matrix)

resultData = {'Survived': train_df['Survived'], 'Pclass': train_df['Pclass'], 'Sex': train_df['Sex'].astype('int32'), 'Age': train_df['Age'].astype('int32'), 'Fare': train_df['Fare'].astype('int32'), 'Embarked': train_df['Embarked'].astype('int32')}
result_df = pd.DataFrame(resultData)
#print(result_df)
X = result_df.drop('Survived', axis=1)
y = result_df['Survived']
SVCClf1 = SVC(kernel='linear', C=1)
scores1 = cross_val_score(SVCClf1, X=X, y=y)
print(scores1.mean())

SVCClf2 = SVC(kernel='poly', C=1)
scores2 = cross_val_score(SVCClf2, X=X, y=y)
print(scores2.mean())

SVCClf3 = SVC(kernel='rbf', C=1)
scores3 = cross_val_score(SVCClf3, X=X, y=y)
print(scores3.mean())


