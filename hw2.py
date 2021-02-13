import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.tree._export import plot_tree
from sklearn.model_selection._validation import cross_val_score
from sklearn import ensemble

#Use your own folder when testing
train_df = pd.read_csv('D:/HWData/train.csv')
test_df = pd.read_csv('D:/HWData/test.csv')

#Task1 Q1 and Q2####################################################################################
fareTrainData = list(filter(lambda x: not pd.isnull(x), train_df['Fare']))
topFare = list(pd.Series(fareTrainData).mode())[0]
print('mode for Fare:', topFare)
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
print('Sex correlation',pd.Series(sexNumData).corr(train_df[train_df.columns[1]]))
train_df.update(sexNumData)

updatedEmbarkedData = train_df['Embarked'].apply(lambda x: 'S' if pd.isnull(x) else x)# use mode to fill null value
train_df.update(updatedEmbarkedData)
updatedEmbarkedData = train_df['Embarked'].apply(lambda x: int(1) if x == 'S' else (int(2) if x == 'Q' else int(3)) )#S=1 Q=2 C=3
train_df.update(updatedEmbarkedData)
print('Embarked correlation',(train_df['Embarked'].astype('int32')).corr(train_df[train_df.columns[1]]))

corr_matrix = train_df.corr(method='pearson')
#print(corr_matrix['Survived']['Pclass'])
print(corr_matrix)

resultData = {'Survived': train_df['Survived'], 'Pclass': train_df['Pclass'], 'Sex': train_df['Sex'].astype('int32'), 'Age': train_df['Age'].astype('int32'),'Embarked': train_df['Embarked'].astype('int32')}
result_df = pd.DataFrame(resultData)
#Task1 Q3##################################################################################################################
resultDecisionTree = DecisionTreeClassifier(criterion='gini')
X = result_df.drop('Survived', axis=1)
y = result_df['Survived']
resultDecisionTree.fit(X, y)
fig = plt.figure(figsize=(35,30))
plot_tree(resultDecisionTree, filled=True)
#plt.show()
#Task1 Q4##################################################################################################################
clf = DecisionTreeClassifier()
scoresforDTC = cross_val_score(clf, X = X, y = y)
print("Average score of DTC:", scoresforDTC.mean())
#Task1 Q5##################################################################################################################
rf = ensemble.RandomForestClassifier()
scoresforRFC = cross_val_score(rf, X = X, y = y)
print("Average score for RFC:", scoresforRFC.mean())

#HW1 As a reference########################################################################################################
# validPassenger=[0 for i in range(len(train_df.columns))]
# #Q7###########################################################
# for i in [5,6,7,9]:#Age, SibSp, Parch, Fare
#     validPassenger[i] = list(filter(lambda x: not pd.isnull(x), train_df[train_df.columns[i]]))
#     print(train_df.columns[i])
#     print('count ', len(validPassenger[i]))
#     print('mean ', sum(validPassenger[i])/len(validPassenger[i]))
#     print('std ', np.std(validPassenger[i]))
#     print('min ', min(validPassenger[i]))
#     print('25% percentile', np.percentile(validPassenger[i], 25))
#     print('50% percentile', np.percentile(validPassenger[i], 50))
#     print('75% percentile', np.percentile(validPassenger[i], 75))
#     print('max ', max(validPassenger[i]))
# #Q8############################################################
# for i in [1, 2, 4, 6, 7, 9, 11]:
#     validPassenger[i] = list(filter(lambda x: not pd.isnull(x), train_df[train_df.columns[i]]))
#     print(train_df.columns[i])
#     print('count',len(validPassenger[i]))
#     print('unique', len(list(set(validPassenger[i]))))
#     top = list(pd.Series(validPassenger[i]).mode())
#     print('top\n', top)
#     freq = list(pd.Series(validPassenger[i]).value_counts())[0:len(top)]
#     print('freq\n', freq)
#     print('')
# #Q9#############################################################
# corr_matrix = train_df.corr(method='pearson')
# print(corr_matrix['Survived']['Pclass'])
# #print(corr_matrix)
# #Q10############################################################
# sexData = list(filter(lambda x: not pd.isnull(x), train_df[train_df.columns[4]]))
# sexNumData = train_df[train_df.columns[4]].apply(lambda x: 1 if x == 'female' else 0)
# print(pd.Series(sexNumData).corr(train_df[train_df.columns[1]]))
# plt.figure()
# notSurvivedBool =  train_df[train_df.columns[1]] == 0
# suvivedBool = train_df[train_df.columns[1]] == 1
# plt.title('Survived = 0')
# train_df[notSurvivedBool][train_df.columns[5]].plot.hist(bins=20)
# plt.figure()
# plt.title('Survived = 1')
# train_df[suvivedBool][train_df.columns[5]].plot.hist(bins=20)
# #Q12#############################################################
# for i in [1, 2, 3]:
#     for j in [0, 1]:
#         plt.figure()
#         pclassSuvived = (train_df['Pclass'] == i) & (train_df['Survived'] == j)
#         plt.title('Pclass =' + str(i) + ' | Survived = ' + str(j))
#         plt.ylim(0,40)
#         train_df[pclassSuvived]['Age'].plot.hist(bins=20)
# #Q13#############################################################
# for i in ['S', 'C', 'Q']:
#     for j in [0, 1]:
#         #plt.figure()
#         embarkedSuvived = (train_df['Embarked'] == i) & (train_df['Survived'] == j)
#         embarkedSuvivedMale = embarkedSuvived & (train_df['Sex'] == 'male')
#         avgMaleFare = pd.DataFrame(train_df[embarkedSuvivedMale]['Fare']).mean()
#         print(avgMaleFare[0])
#         embarkedSuvivedFemale = embarkedSuvived & (train_df['Sex'] == 'female')
#         avgFemaleFare = pd.DataFrame(train_df[embarkedSuvivedFemale]['Fare']).mean()
#         print(avgFemaleFare[0])
#          
#         tempdf = pd.DataFrame({'Sex':['female', 'male'], 'Fare': [avgFemaleFare[0] ,avgMaleFare[0]]})
#         tempdf.plot.bar(x='Sex', y='Fare', rot = 0)
#         plt.ylim(0, 80)
#         plt.title('Embarked =' + str(i) + ' | Survived = ' + str(j))
# #Q14#############################################################
# #Fare
# duplicatedBool = train_df[train_df.columns[8]].duplicated()
# #duplicate rate
# print('duplicate rate:', len(train_df[duplicatedBool])/len(train_df))
# #Q15#############################################################
# cabinNullTrainData = list(filter(lambda x: pd.isnull(x), train_df['Cabin']))
# cabinNullTestData = list(filter(lambda x: pd.isnull(x), test_df['Cabin']))
# print('Cabin Null Values', len(cabinNullTrainData)+len(cabinNullTestData))
# #Q16#############################################################
# sexData = list(filter(lambda x: not pd.isnull(x), train_df[train_df.columns[4]]))
# sexNumData = train_df[train_df.columns[4]].apply(lambda x: 1 if x == 'female' else 0)
# print('correlation between Sex and Survived:', pd.Series(sexNumData).corr(train_df[train_df.columns[1]]))
# #Q18#############################################################
# updatedEmbarkedData = train_df['Embarked'].apply(lambda x: 'S' if pd.isnull(x) else x)
# train_df.update(updatedEmbarkedData)
# #Q19#############################################################
# fareTestData = list(filter(lambda x: not pd.isnull(x), test_df['Fare']))
# topFare = list(pd.Series(fareTestData).mode())[0]
# print('mode for Fare:', topFare)
# updatedFareData = train_df['Fare'].apply(lambda x: topFare if pd.isnull(x) else x)
# train_df.update(updatedFareData)
# #Q20#############################################################
# updatedFareData = train_df['Fare'].apply(lambda x: 0 if x>-0.001 and x<=7.91 else (1 if x>7.91 and x<=14.454 else (2 if x>14.454 and x<=31 else 3)))
# #print(updatedFareData)
# train_df.update(updatedFareData)
# plt.show()#uncomment this line when you need testing