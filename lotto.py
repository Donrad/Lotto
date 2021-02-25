import requests
import urllib.request
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import io
import csv
import random
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from math import sqrt

test_years = []
train_years = []

all_bonus_balls_train = {}		
all_draws_train = {}
all_bonus_balls_test = {}		
all_draws_test = {}
all_21 = {}
all_21_bonus_balls = {}

train_df = []
test_df = []
test_21_df = []

def split_overlap(array,size,overlap):
    result = []
    while True:
        if len(array) <= size:
            result.append(array)
            return result
        else:
            result.append(array[:size])
            array = array[size-overlap:]

for year in range(1994, 2017):
	train_years.append(year)

for year in range(2017, 2021):
	test_years.append(year)

def getTrainData():
	for year in train_years:
		headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.53'}
		url = f'https://www.national-lottery.com/lotto/results/{year}-archive'
		r = requests.get(url, headers=headers)
		soup = BeautifulSoup(r.text, 'html.parser')
		uls = soup.find_all('ul', {'class' : 'balls'})

		for count_of_uls, each_ul in enumerate(uls):
			each_dict = {}

			for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark ball')):
				each_dict[str(count_of_lis)] = num.text

			all_draws_train[str(count_of_uls)] = each_dict

		for count_of_uls, each_ul in enumerate(uls):
			each_dict = {}

			for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark bonus-ball')):
				each_dict['bonus-ball'] = num.text

			all_bonus_balls_train[str(count_of_uls)] = each_dict

		ditems = all_draws_train.items()
		bitems = all_bonus_balls_train.items()

		ddata = list(ditems)
		bdata = list(bitems)

		darr = np.array(ddata)
		barr = np.array(bdata)

		x = np.append(darr, barr, axis=1)

		N = [int(v) for d in x.ravel()[1::2] for v in d.values()]

		split = split_overlap(N,7,0)
		
		for array in split:
			train_df.append(array)

#		strint = [int(i) for i in str(x[-1][3:4]).split() if i.isdigit()]


def getTestData():
	for year in test_years:
		headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.53'}
		url = f'https://www.national-lottery.com/lotto/results/{year}-archive'
		r = requests.get(url, headers=headers)
		soup = BeautifulSoup(r.text, 'html.parser')
		uls = soup.find_all('ul', {'class' : 'balls'})

		for count_of_uls, each_ul in enumerate(uls):
			each_dict = {}

			for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark ball')):
				each_dict[str(count_of_lis)] = num.text

			all_draws_test[str(count_of_uls)] = each_dict

		for count_of_uls, each_ul in enumerate(uls):
			each_dict = {}

			for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark bonus-ball')):
				each_dict['bonus-ball'] = num.text

			all_bonus_balls_test[str(count_of_uls)] = each_dict

		ditems = all_draws_test.items()
		bitems = all_bonus_balls_test.items()

		ddata = list(ditems)
		bdata = list(bitems)

		darr = np.array(ddata)
		barr = np.array(bdata)

		x = np.append(darr, barr, axis=1)

		N = [int(v) for d in x.ravel()[1::2] for v in d.values()]

		split = split_overlap(N,7,0)
		
		for array in split:
			test_df.append(array)

def get21TestData():
	headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.53'}
	url = f'https://www.national-lottery.com/lotto/results/2021-archive'
	r = requests.get(url, headers=headers)
	soup = BeautifulSoup(r.text, 'html.parser')
	uls = soup.find_all('ul', {'class' : 'balls'})

	for count_of_uls, each_ul in enumerate(uls):
		each_dict = {}

		for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark ball')):
			each_dict[str(count_of_lis)] = num.text

		all_21[str(count_of_uls)] = each_dict

	for count_of_uls, each_ul in enumerate(uls):
		each_dict = {}

		for count_of_lis, num in enumerate(each_ul.find_all('li', class_='result medium lotto ball dark bonus-ball')):
			each_dict['bonus-ball'] = num.text

		all_21_bonus_balls[str(count_of_uls)] = each_dict

	ditems = all_21.items()
	bitems = all_21_bonus_balls.items()

	ddata = list(ditems)
	bdata = list(bitems)

	darr = np.array(ddata)
	barr = np.array(bdata)

	x = np.append(darr, barr, axis=1)

	N = [int(v) for d in x.ravel()[1::2] for v in d.values()]

	split = split_overlap(N,7,0)
	
	for array in split:
		test_21_df.append(array)

#-----------------------------MODEL--------------------------------

train_data = train_df
test_data = test_df + test_21_df
all_data = train_data + test_data

#------------------Separate First Number in Arrays-----------------

train_count = []
test_count = []
all_count = []

train_ini =[]
test_ini = []
all_ini = []

def ft_setup():

	makedatacsv = open('data.csv', 'w')

	getTrainData()
	getTestData()
	get21TestData()

	train_data = train_df
	test_data = test_df + test_21_df
	all_data = train_data + test_data

	for i in range(0, len(train_data)):
		train_count.append(i)

	for i in range(0, len(test_data)):
		test_count.append(i)

	for i in range(0, len(all_data)):
		all_count.append(i)

	index = 0

	for count in train_count:
		initial_draw = train_data[index][0]
		train_ini.append(initial_draw)	
		index += 1

	index = 0

	for count in test_count:
		initial_draw = test_data[index][0]
		test_ini.append(initial_draw)
		index += 1

	index = 0

	for count in all_count:
		initial_draw = all_data[index][0]
		all_ini.append(initial_draw)
		index += 1

	newarr = []

	for array in all_data:
		a0 = array[0]
		a1 = array[1]
		a2 = array[2]
		a3 = array[3]
		a4 = array[4]
		a5 = array[5]
		sx = sum(array)
		del array[6]

		array.append(sx)

		sub_index_no_bb = [a0, a1-a0, a2-a1, a3-a2, a4-a3, a5-a4] #Shorthand option - no mathematical benefit for prediciton, but possibly for computation.
		newarr.append(array)

	def write_data():

		labels = ['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Sum']

		with open('data.csv', 'w', newline='') as spread:
			thewriter = csv.writer(spread)
			thewriter.writerow(labels)
			for d in newarr:
				thewriter.writerow(d)

	write_data()

#------------------------------------------------------------------

# Draw mean

n_m = {
	0: 7.687547,
	1: 15.350418,
	2: 22.831815,
	3: 30.268033,
	4: 37.421792,
	5: 44.91381,
	6: 155,
}

#------------------------------------------------------------------

#ifa = []

#for i in random.sample(range(1, 7), 6):
#	ifa.append(i)

#print(ifa)

#n_i_a = sorted(ifa)

#n_i = {
#	0: n_i_a[0],
#	1: n_i_a[1],
#	2: n_i_a[2],
#	3: n_i_a[3],
#	4: n_i_a[4],
#	5: n_i_a[5],
#	6: sum(n_i_a),
#}


#print(n_i)

#------------------------------------------------------------------

try:
	df = pd.read_csv("data.csv")
	df.astype(np.float64)
except FileNotFoundError:
	pass

n_i = {
	0: random.randint(1, 16),
	1: 0,
	2: 1,
	3: 0,
	4: 0,
	5: 0,
	6: 0,
}

predidex_full = [1, 2, 3, 4, 5]

tree_model = DecisionTreeRegressor()
rf_model = RandomForestClassifier()
scaler = StandardScaler()
rfp = rf_model

#--------------------------------------MAIN FUNCS------------------------------------------

def display(results):
	print(f'Best parameters are: {results.best_params_}')
	print("\n")
	mean_score = results.cv_results_['mean_test_score']
	std_score = results.cv_results_['std_test_score']
	params = results.cv_results_['params']

	for mean,std,params in zip(mean_score,std_score,params):
		print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


def train():

	print("*" *48)
	print("-" *16 + " TRAINING MODEL " + "-" *16)
	print("_" *48)
	print("\n")

	try:
		df = pd.read_csv("data.csv")
		df.astype(np.float64)
	except FileNotFoundError:
		pass

	X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Sum']]
	y = df['Draw 5']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100000)

	scaler = StandardScaler()
	train_scaled = scaler.fit_transform(X_train)
	test_scaled = scaler.transform(X_test)

	tree_model = DecisionTreeRegressor()
	rf_model = RandomForestClassifier()

	tree_model.fit(train_scaled, y_train)
	rf_model.fit(train_scaled, y_train)

	tree_mse = mean_squared_error(y_train, tree_model.predict(train_scaled))
	tree_mae = mean_absolute_error(y_train, tree_model.predict(train_scaled))
	rf_mse = mean_squared_error(y_train, rf_model.predict(train_scaled))
	rf_mae = mean_absolute_error(y_train, rf_model.predict(train_scaled))

	print("Decision Tree training mse = ",tree_mse," & mae = ",tree_mae," & rmse = ", sqrt(tree_mse))
	print("Random Forest training mse = ",rf_mse," & mae = ",rf_mae," & rmse = ", sqrt(rf_mse))

	tree_test_mse = mean_squared_error(y_test, tree_model.predict(test_scaled))
	tree_test_mae = mean_absolute_error(y_test, tree_model.predict(test_scaled))
	rf_test_mse = mean_squared_error(y_test, rf_model.predict(test_scaled))
	rf_test_mae = mean_absolute_error(y_test, rf_model.predict(test_scaled))

	print("Decision Tree test mse = ",tree_test_mse," & mae = ",tree_test_mae," & rmse = ", sqrt(tree_test_mse))
	print("Random Forest test mse = ",rf_test_mse," & mae = ",rf_test_mae," & rmse = ", sqrt(rf_test_mse))

	parameters = {
	    "n_estimators":[10],
	    "max_depth":[None],
	    "min_samples_leaf":[1]
	}

	cv = GridSearchCV(rf_model,parameters,cv=5)
	cv.fit(X,y.ravel())

	print()
	print('Determining optimal parameters...')
	print()

	display(cv)

	print()
	print("*" *48)
	print("-" *19 + " FINISHED " + "-" *19)
	print("_" *48)
	print("\n")


def rf_predict_func(test_dat):
	rfp = rf_model.predict(test_dat.reshape(1, -1))
	n_i[6] = n_i[0] + n_i[1] + n_i[2] + n_i[3] + n_i[4] + n_i[5]
	return int(rfp)

def predict5():

	try:
		df = pd.read_csv("data.csv")
		df.astype(np.float64)
	except FileNotFoundError:
		pass

	print("-" * 48)
	print("Enter your lucky number (recommended between 1-10)")
	print("\n")
	option = input("First Number: ")
	n_i[0] = int(option)

	for predidex in predidex_full:
		rfp = 0 
		if predidex == 1:
			X = df[['Draw 0', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 1']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), round(float(n_i[2])), round(float(n_i[3])), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[1] = rfp
		elif predidex == 2:
			X = df[['Draw 0', 'Draw 1', 'Draw 3', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 2']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), round(float(n_i[3])), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[2] = rfp
		elif predidex == 3:
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 3']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[3] = rfp
		elif predidex == 4:
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 5', 'Sum']]
			y = df['Draw 4']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), int(n_i[3]), round(float(n_i[5])), int(n_i[6])]))
			n_i[4] = rfp
		elif predidex == 5:	
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Sum']]
			y = df['Draw 5']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), int(n_i[3]), int(n_i[4]), int(n_i[6])]))
			n_i[5] = rfp
		else:
			print('Predidex Axis Iteration Error')

	print("\n")
	print("-" * 60)
	print("The numbers are predicted to be: " + f'{n_i[0]} | {n_i[1]} | {n_i[2]} | {n_i[3]} | {n_i[4]} | {n_i[5]}')
	print("-" * 60)
	print("\n")

def predict6():

	try:
		df = pd.read_csv("data.csv")
		df.astype(np.float64)
	except FileNotFoundError:
		pass

	print("-" * 48)
	print("\n")

	for predidex in predidex_full:
		if predidex == 1:
			X = df[['Draw 0', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 1']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), round(float(n_i[2])), round(float(n_i[3])), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[1] = rfp
		elif predidex == 2:
			X = df[['Draw 0', 'Draw 1', 'Draw 3', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 2']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), round(float(n_i[3])), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[2] = rfp
		elif predidex == 3:
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 4', 'Draw 5', 'Sum']]
			y = df['Draw 3']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), round(float(n_i[4])), round(float(n_i[5])), int(n_i[6])]))
			n_i[3] = rfp
		elif predidex == 4:
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 5', 'Sum']]
			y = df['Draw 4']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), int(n_i[3]), round(float(n_i[5])), int(n_i[6])]))
			n_i[4] = rfp
		elif predidex == 5:	
			X = df[['Draw 0', 'Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Sum']]
			y = df['Draw 5']
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
			train_scaled = scaler.fit_transform(X_train)
			test_scaled = scaler.transform(X_test)
			tree_model.fit(train_scaled, y_train)
			rf_model.fit(train_scaled, y_train)
			rfp = rf_predict_func(np.array([int(n_i[0]), int(n_i[1]), int(n_i[2]), int(n_i[3]), int(n_i[4]), int(n_i[6])]))
			n_i[5] = rfp
		else:
			print('Predidex Axis Iteration Error')

	print("\n")
	print("-" * 60)
	print("The numbers are predicted to be: " + f'{n_i[0]} | {n_i[1]} | {n_i[2]} | {n_i[3]} | {n_i[4]} | {n_i[5]}')
	print("-" * 60)
	print("\n")

#---------------------------------SYSTEM------------------------------------

def run_menu():
	print("*" *48)
	print("-" *10 + " What would you like to do? " + "-" *10)
	print("\n")
	print("1. Predict 5 numbers from my lucky number")
	print("2. Predict 6 numbers")
	print("3. First time setup")
	print("\n")

	option = int(input("Enter option: "))

	if option == 1 or option == 2:
		run_program(option)
	elif option == 3:
		print('\nInitialising, please wait.\n')
		ft_setup()
		print('\nComplete\n')

		df = pd.read_csv('data.csv')

		run_menu()
	elif option == 5:
		train()
		run_menu()
	elif option == 9:
		exit()
	else:
		print('#---------------------------Invalid---------------------------#')
		print('| 1 & 2: Predict Numbers | 3: Re-train Model | 9: Exit Script |')
		return option

def run_program(option):
	if option == 1:
		predict5()
	elif option == 2:
		predict6()

if __name__== "__main__":
	run_menu()

	try:
		df = pd.read_csv("data.csv")
		df.describe()
		df.astype(np.float64)
	except FileNotFoundError:
		pass