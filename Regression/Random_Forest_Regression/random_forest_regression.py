# Random Forest Regression

# Importing the libraries
import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV,cross_val_score

# load dataset
dataframe = pandas.read_csv("lending_club_loans_Clean_MAI_Light_Demo.csv",na_values = ['Missing','NA','na','N/A','n/a',''],encoding = "ISO-8859-1")
dataframe.replace(["NaN", 'NaT','','Missing','NA','na','N/A','n/a','nan','NAN'], numpy.nan, inplace = True)
dataframe.fillna(0,inplace = True)
dataframe.drop(dataframe.columns[[0,1,2]], axis=1, inplace = True)
dataframe[['int_rate','revol_util']]=dataframe[['int_rate','revol_util']].replace('%','',regex=True).astype(float).div(100)

data_num = dataframe._get_numeric_data()
data_cat = dataframe.select_dtypes(exclude=["number"])
data_cat.drop(['initial_list_status','title','pymnt_plan','emp_title', 'issue_d', 'zip_code','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d','application_type','addr_state'], axis=1, inplace=True)

#Categorical data preparation:
dummy1 = pandas.get_dummies(data_cat['term'])
del dummy1[' 60 months']
del data_cat['term']
dum_1 = pandas.concat([data_cat, dummy1], axis=1)
# print(dum_1)

dummy2 = pandas.get_dummies(data_cat['grade'])
del dummy2['G']
del dum_1['grade']
dum_2 = pandas.concat([dum_1, dummy2], axis=1)
# print(dum_2)

# print(data_cat['sub_grade'].value_counts())
dummy3 = pandas.get_dummies(data_cat['sub_grade'])
del dummy3['G5']
del dum_2['sub_grade']
dum_3 = pandas.concat([dum_2, dummy3], axis=1)
# print(dum_3)

# print(data_cat['emp_length'].value_counts())
dummy4 = pandas.get_dummies(data_cat['emp_length'])
del dummy4[0]
del dum_3['emp_length']
dum_4 = pandas.concat([dum_3, dummy4], axis=1)
# print(dum_4.columns.values)

# print(data_cat['home_ownership'].value_counts())
dummy5 = pandas.get_dummies(data_cat['home_ownership'])
# print(dummy5)
del dummy5['OWN']
del dum_4['home_ownership']
dum_5 = pandas.concat([dum_4, dummy5], axis=1)
# print(dum_5.columns.values)

# print(data_cat['verification_status'].value_counts())
dummy6 = pandas.get_dummies(data_cat['verification_status'])
# print(dummy6)
del dummy6['Source Verified']
del dum_5['verification_status']
dum_6 = pandas.concat([dum_5, dummy6], axis=1)
# print(dum_6.columns.values)

# print(data_cat['loan_status'].value_counts())
dummy7 = pandas.get_dummies(data_cat['loan_status'])
del dummy7['Late (31-120 days)']
del dum_6['loan_status']
dum_7 = pandas.concat([dum_6, dummy7], axis=1)
# print(dum_7.columns.values)

# print(data_cat['purpose'].value_counts())
dummy8 = pandas.get_dummies(data_cat['purpose'])
del dummy8['renewable_energy']
del dum_7['purpose']
dum_8 = pandas.concat([dum_7, dummy8], axis=1)
# print(dum_8.shape)

df = pandas.concat([data_num,dum_8], axis=1)

y = df['funded_amnt']
X = df[['int_rate', 'installment', 'annual_inc',
	 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
	 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
	 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
	 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
	 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
	 'last_pymnt_amnt', 'last_fico_range_high', 'last_fico_range_low',
	 ' 36 months', 'A', 'B', 'C', 'D', 'E', 'F', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2',
	 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1',
	 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', '1 year',
	 '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
	 '8 years', '9 years', '< 1 year', 'MORTGAGE', 'RENT', 'Not Verified',
	 'Verified', 'Charged Off', 'Current', 'Fully Paid', 'In Grace Period', 'car',
	 'credit_card', 'debt_consolidation', 'home_improvement', 'house',
	 'major_purchase', 'medical', 'moving', 'other', 'small_business', 'vacation', 'wedding']]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 10,
								random_state = 0,
								max_features = 'auto',
								min_samples_leaf = 10)
model = regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = model.predict(X_test)

#Evaluating mean squared error---1.Validation metric
def mse():
	forest_mse = mean_squared_error(y_pred, y_test)
	print('Mean squared error : ',forest_mse)

#Evaluating root mean squared error---2. Validation metric
def rmse():
	forest_mse = mean_squared_error(y_pred, y_test)
	forest_rmse = numpy.sqrt(forest_mse)
	print('Root mean squared error : ',forest_rmse)

y_pred_df = pandas.DataFrame(y_pred)
y_pred_df = y_pred_df.rename(columns={0: 'Predicted_val'})

#Appending actual and predicted values inside a single dataframe
# df1 = df.join(y_pred_df)

#Evaluating R-squared---3. Validation metric
def r_squared():
	r2 = r2_score(y_test, y_pred)
	print('R-squared value : ',r2)
	# SSres = numpy.square(df1['funded_amnt'] - df1['Predicted_val'])
	# SStot = numpy.square(df1['funded_amnt'] - df1['Predicted_val'].mean())
	# r2 = 1 - (SSres.sum()/(SStot.sum()))
	# print(r2)

#Evaluating MAPE---4. Validation metric
def mape_avg():
	y_pred = model.predict(X_test)
	y_true, y_pred = numpy.array(y_test), numpy.array(y_pred)
	mape = numpy.mean(numpy.abs((y_true - y_pred) / y_true)) * 100
	print('Mean average percentage in aggregate : ',mape)

#Evaluating feature importances---5. Validation metric
def feature_importance():
	feature_labels = numpy.array(['int_rate', 'installment', 'annual_inc',
		 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
		 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec',
		 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
		 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
		 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
		 'last_pymnt_amnt', 'last_fico_range_high', 'last_fico_range_low',
		 ' 36 months', 'A', 'B', 'C', 'D', 'E', 'F', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2',
		 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1',
		 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', '1 year',
		 '10+ years', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
		 '8 years', '9 years', '< 1 year', 'MORTGAGE', 'RENT', 'Not Verified',
		 'Verified', 'Charged Off', 'Current', 'Fully Paid', 'In Grace Period', 'car',
		 'credit_card', 'debt_consolidation', 'home_improvement', 'house',
		 'major_purchase', 'medical', 'moving', 'other', 'small_business', 'vacation', 'wedding'])
	importance = model.feature_importances_
	feature_indexes_by_importance = importance.argsort()
	impact_test = []
	for index in feature_indexes_by_importance:
	    impact = (feature_labels[index], (importance[index] * 100.0))
	    impact_test.append(impact)
	print('Variable importance : ',impact_test)


#implementing K-fold cross validation---
def k_fold():
    accuracies = cross_val_score(estimator = model,
                                X = X_train,
                                y = y_train,
                                cv = 10,
                                n_jobs = -1)
    print('K-fold accuracies : ',accuracies)
    print('K-fold avg accuracy : ',accuracies.mean())
    print('K-fold accuracy std deviation : ',accuracies.std())


#Applying Grid-search and find best models and parameters---
def grid_search():
	param_grid = { 
    'n_estimators': [10,50,100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [10,50,100]}
	CV_rfc = GridSearchCV(estimator = model,
							param_grid = param_grid,
							cv= 5,
							n_jobs = -1)
	CV_rfc.fit(X, y)
	print('Grid-search best parameters :')
	print (CV_rfc.best_params_)
	print('grid_search best score :')
	print(CV_rfc.best_score_)

mse()
rmse()
r_squared()
mape_avg()
feature_importance()
k_fold()
grid_search()