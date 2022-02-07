#regression based ML project

#Dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot #for plots
import seaborn as sns #for graphs
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


house_price_dataset = sklearn.datasets.load_boston()
#print(house_price_dataset)

#Loading the Dataset to pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data ,columns=house_price_dataset.feature_names) #here columns add the title of every field.

# adding house price to the list
house_price_dataframe["price"]=house_price_dataset.target
#print(house_price_dataframe)

#checking number of rows and columns
#print(house_price_dataframe.shape())

#checking for missing values
#print(house_price_dataframe.isnull().sum())

# statistical measures of the dataset
#print(house_price_dataframe.describe())


#understanding corelation between data

correlation = house_price_dataframe.corr()

#constructing a heat map to understand the correlation
plt.Figure(figsize=(10,10))
#print(sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')) #cbar= colorbar, .1f= flot value after decimal and column names= annot

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']
#print(X)
#print(Y)

#spliting data into test and training data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2) #20% of total is in test size

#print(X.shape, x_train.shape, x_test.shape)

#Model Training
#XGBoost Regressor

# loading the model
model = XGBRegressor()


# training the model with X_train
model.fit(x_train, y_train)

#Prediction on training data
# accuracy for prediction on training data
training_data_prediction = model.predict(x_train)
#print(training_data_prediction)


#ERRORS IN PREDICTIONS
# R squared error
error_1 = metrics.r2_score(y_train, training_data_prediction)

# Mean Absolute Error
error_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print("R squared error : ", error_1)
print('Mean Absolute Error : ', error_2)

#Visualizing the actual Prices and predicted prices
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Preicted Price")
plt.show()

#Prediction on Test Data
# accuracy for prediction on test data
test_data_prediction = model.predict(x_test)
score_1 = metrics.r2_score(x_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(y_test, test_data_prediction)

print("R squared error : ", score_1)
print('Mean Absolute Error : ', score_2)
