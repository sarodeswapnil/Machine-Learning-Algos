#!/home/mban/anaconda2/bin/python
import numpy as np
import random
import matplotlib.pyplot as plt
#the decision tree model we will train and evaluate
from sklearn import tree 

#define function to process record in file
def processText(data):
	data = data.strip()
	temp_data = []	
	fields = data.split(',')
	for field in fields:
		temp_data.append(field)
	return temp_data 

# train a decision tree classifier and create a new model with
# the specified metaparameters
def build_tree(trainingX, trainingY, testX, testY):
	range_depth = [0,2,4,8,16]
	range_nodes = [2,4,8,16,32,64,128,256]
	for max_tree_depth in range_depth:	
		accuracy = []
		for max_nodes in range_nodes:
			# a varaiable for max_leaf_nodes so we can print it out, 
			if max_tree_depth == 0:
				#for default max depth of the tree
				clf = tree.DecisionTreeClassifier \
					(max_leaf_nodes=max_nodes)
			else:
				#for specific values of maximum nodes 
				#and max depth of tree
				clf = tree.DecisionTreeClassifier \
					(max_leaf_nodes=max_nodes, \
					 max_depth=max_tree_depth)

			#train the model (fit the model to the data)
			clf.fit(trainingX, trainingY)

			#count number of correct and incorrect predictions
			#starting at 0
			correct = 0
			incorrect = 0
			#use the model to make predictions for the testing input vector
			predictions = clf.predict(testX)

			#evaluate the predictions against the testing target bector
			for i in range(0, predictions.shape[0]):
				if (predictions[i] == testY[i]):
					correct += 1
				else:
					incorrect += 1
			#compute accuracy
			accuracy.append(float(correct)/(correct+incorrect))
		#plot accuracy of decision tree against the number of nodes
		plt.plot(range_nodes,accuracy, color='black')
		plt.xticks(range_nodes)
		plt.title(["Max Depth = ",'%f'%(max_tree_depth)])	

		# Label for x-axis
		plt.xlabel("Number of nodes")
		# Label for y-axis	
		plt.ylabel("Accuracy")
	
		plt.show()
		print "Enter to continue..."
		raw_input()

#Define function to read and parse row of input data
def get_row(i,j,input_data):
	newRow=[]
	for j in range(len(input_data[i])-1):
		#assign numerical classes for data with strings
		if j==1:
			if input_data[i][j]=='F':
				newRow.append(0)
			else:
				newRow.append(1)
		elif j==3:
			if input_data[i][j]=='Single':
				newRow.append(0)
			else:
				newRow.append(1)
		elif j==4:
			if input_data[i][j]=='Low':
				newRow.append(0)
			elif input_data[i][j]=='Medium':
				newRow.append(1)
			elif input_data[i][j]=='Heavy':
				newRow.append(2)
			else:
				newRow.append(3)
		elif j==5:
			if input_data[i][j]=='Automatic':
				newRow.append(0)
			else:
				newRow.append(1)
		elif j==6:
			if input_data[i][j]=='12 months':
				newRow.append(0)
			elif input_data[i][j]=='36 months':
				newRow.append(1)
			else:
				newRow.append(2)
		elif j==7:
			if input_data[i][j]=='N':
				newRow.append(0)
			else:
				newRow.append(1)
		elif j==8:
			if input_data[i][j]=='N':
				newRow.append(0)
			else:
				newRow.append(1)
		else:				
			newRow.append(int(input_data[i][j]))
	return newRow

input_data=[]		

fileName = "training_set.csv"
#open file in read-only mode
file = open(fileName, "r") 
i=0

#read one line from the file at a time..
for line in file:
	input_data.append(processText(line))
j=0	

#starting with empty, ordinary python lists for-
#training input vector
trainingX = []
#training target vector - correct class labels
trainingY = []
#testing set input vector
testX = []
#testing target vector - correct class labels
testY = []
num_ones = 0
#loops to construct the above lists from the original dataset
for i in range(len(input_data)):
	newRow=[]
	#construct a new row as a list from the first two columns of the 
	#current row (the row at index "i")
	newRow=get_row(i,j,input_data)

	#we want to put 10% into a testing set that is not use to to train the model
	if(i%10==0): 
		#put every tenth row into the test set
		testX.append(newRow)
		#the test set needs to be constructed from the corresponding target variables
		if 'Late' in input_data[i][len(newRow)]:
			testY.append(1)
			num_ones+=1
		else:
			testY.append(0)
	else:
		#put into the training set
		trainingX.append(newRow)
		if 'Late' in input_data[i][len(newRow)]:
			trainingY.append(1)
			num_ones+=1
		else:
			trainingY.append(0)

#call function to build tree and plot graphs
build_tree(np.array(trainingX), np.array(trainingY), np.array(testX), np.array(testY))

#training input vector
training_bal_X = []
#training target vector 
training_bal_Y = []
#testing set input vector
test_bal_X = []
#testing target vector
test_bal_Y = []
total_x=[]
total_y=[]
total_x.extend(trainingX)
total_x.extend(testX)
total_y.extend(trainingY)
total_y.extend(testY)
total_data=total_x
for i in range(len(total_x)):
	total_data[i].append(total_y[i])	
num_zeroes = len(total_data)-num_ones
#sort data by target to ensure data has all observations with target 0
#followed by all observations with target 1
total_data = sorted(total_data,key=lambda total_data:total_data[9])
#for a balanced dataset, identify which result is more in the target - 
#0s or 1s
if num_ones > num_zeroes:
	for i in range(num_zeroes):
		#put every 10th observation in test data, else training data,
		#to have 90% in training data and 10% in test data
		if(i%10==0): 
			test_bal_X.append(total_data[i][:8])		
			test_bal_Y.append(total_data[i][9])		
		else:
			training_bal_X.append(total_data[i][:8])
			training_bal_Y.append(total_data[i][9])
	#select random numbers to decide which observation goes to the
	# training and test data		
	#to_select = random.sample(range(num_ones), num_zeroes)
	to_select = range(num_zeroes)
	for i in range(num_ones):
		if(i%10==0): 
			if i in to_select:			
				test_bal_X.append(total_data[i+num_zeroes][:8])		
				test_bal_Y.append(total_data[i+num_zeroes][9])		
		else:
			if i in to_select:			
				training_bal_X.append(total_data[i+num_zeroes][:8])
				training_bal_Y.append(total_data[i+num_zeroes][9])		
else:
	#to_select = random.sample(range(num_zeroes), num_ones)
	to_select = range(num_ones)
	for i in range(num_zeroes):
		if(i%10==0):
			if i in to_select:			
				test_bal_X.append(total_data[i][:8])		
				test_bal_Y.append(total_data[i][9])		 
		else:
			if i in to_select:			
				training_bal_X.append(total_data[i][:8])
				training_bal_Y.append(total_data[i][9])		
	for i in range(num_ones):
		if(i%10==0):
				test_bal_X.append(total_data[i+num_zeroes][:8])		
				test_bal_Y.append(total_data[i+num_zeroes][9])		
		else:
				training_bal_X.append(total_data[i+num_zeroes][:8])
				training_bal_Y.append(total_data[i+num_zeroes][9])		

#call function to build tree and plot graphs for balanced data
build_tree(np.array(training_bal_X), np.array(training_bal_Y), \
	np.array(test_bal_X), np.array(test_bal_Y))

