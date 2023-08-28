
## Libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# Dataframe
df_orig = pd.read_csv(r'C:\Users\Vindo Singh\Desktop\Revision\Titanic Machine Learning Model\train.csv')
'''
print(df_orig)
print(df_orig.info())
print(df_orig.shape)
print(df_orig.size)
print(df_orig.isnull().sum())
'''


# Now using Data-driven Imputation to fill the missing values
df = df_orig[['Survived', 'Pclass','Age', 'SibSp', 'Parch' , 'Fare']]

complete_data = df[df['Age'].notnull()]
incomplete_data = df[df['Age'].isnull()]
'''
print(complete_data)
print(incomplete_data)
'''

x_train = complete_data.drop('Age', axis = 1)
y_train = complete_data['Age']

x_test = incomplete_data.drop('Age',axis = 1)

Model = LinearRegression()
Model.fit(x_train,y_train)

pred_age  = Model.predict(x_test)

df.loc[df['Age'].isnull(), 'Age'] = pred_age


# Dropping Age column from  df_orig
df_orig.drop(['Age'],axis=1, inplace=True)
df_orig.drop(['Cabin'],axis = 1, inplace = True)
df_orig.drop(['Name'], axis = 1, inplace =True)
df_orig.drop(['Ticket'], axis = 1, inplace =True)

df_orig['Age'] = df['Age']
df_orig.dropna(subset=['Embarked'], inplace=True)


# converting  'Sex' column and 'Embarked' into integer form
df_1 = pd.get_dummies(df_orig, columns=['Sex','Embarked'], drop_first=True)
'''
print(df_1.info())
'''

# input and output

x = df_1.iloc[:,2:].values
y = df_1.iloc[:,1].values

'''
print(x,y)
'''

# run regressor

titanic_model = LinearRegression()

titanic_model.fit(x,y)






















