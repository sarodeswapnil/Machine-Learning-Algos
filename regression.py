from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import math

#Load dataset
boston = datasets.load_boston()
#Get independent observation values
xvalues = boston.data
#Get observed y values
yvalues = boston.target


#Define function to calculate y values 
def model(b,x):
	y=0.0
	# y = b0 + b1*x1 + b2*x2 + ... + bn*xn	
	for i in range(0,len(x)):
		y=y+(b[i+1]*x[i])
	y = b[0] + y
	#return y means that y is the computed value that we're 
	#returning to the caller                   
	return y 

#Define function to calculate root mean square error
def rmse_calc(predicted,observed,num_obs):
	rmse = 0.0
	for i in range(num_obs):
		rmse = (rmse + (predicted[i]-observed[i])**2)
	return ((rmse/num_obs)**(0.5))
	#return rmseval
 
#create list of learning rates
learning_rate = np.geomspace(0.00001,0.1,num=5,endpoint=True)
training_data_x = []
training_data_y = []
holdout_data_x= []
holdout_data_y = []

#create lists of min and max for normalization
hold_min = [100000000]*xvalues.shape[1]
hold_max=[-1000000000]*xvalues.shape[1]
train_min = [100000000]*xvalues.shape[1]
train_max=[-1000000000]*xvalues.shape[1]

num_hold = 0
num_train = 0
for i in range(xvalues.shape[0]):
	#To divide data into training set (90%) and holdout set (10%),
	#put every tenth element into the holdout set
	if i%10==0:
		#add to holdout data
		holdout_data_x.append([])
		for j in range(xvalues.shape[1]):
			holdout_data_x[num_hold].append(xvalues[i][j])
			if holdout_data_x[num_hold][j] < hold_min[j]:
				hold_min[j]=holdout_data_x[num_hold][j]
			elif holdout_data_x[num_hold][j] > hold_max[j]:
				hold_max[j]=holdout_data_x[num_hold][j]
		holdout_data_y.append(yvalues[i])
		num_hold+=1
	else:
		#add to training data
		training_data_x.append([])
		for j in range(xvalues.shape[1]):
			training_data_x[num_train].append(xvalues[i][j])
			if training_data_x[num_train][j] < train_min[j]:
				train_min[j]=training_data_x[num_train][j]
			elif training_data_x[num_train][j] > train_max[j]:
				train_max[j]=training_data_x[num_train][j]
		training_data_y.append(yvalues[i])
		num_train+=1

#normalize data with min = 0 and max = 1
for i in range(0,num_hold):
	for j in range(xvalues.shape[1]):
		holdout_data_x[i][j]=(holdout_data_x[i][j] - \
		hold_min[j])/(hold_max[j]-hold_min[j])

for i in range(num_train):
	for j in range(0,xvalues.shape[1]):
		training_data_x[i][j]=(training_data_x[i][j] - \
		train_min[j])/(train_max[j]-train_min[j])

#number of iterations to be made for each learning rate
num_epochs = 10

for rate_num in range(len(learning_rate)):
	b = [0.0]*(xvalues.shape[1]+1)	
	rms_error = [0]*num_epochs
	for epoch in range(num_epochs):
		for i in range(num_train):
			#calculate error for the observation as 
			#difference of predicted and observed value
			error = model(b, training_data_x[i]) - \
				training_data_y[i]
			b[0]= b[0] - learning_rate[rate_num]*error 
			for j in range(1,len(training_data_x[i])):
				b[j]= b[j] - learning_rate[rate_num]* \
				error*training_data_x[i][j-1]

		y_predictions = [0]*num_hold
		for i in range(num_hold):
			#compute predictions and plot to see the result
			y_predictions[i]=model(b,holdout_data_x[i])
		rms_error[epoch]=rmse_calc(y_predictions,holdout_data_y,num_hold)
		
	#plot graph of root mean square error for each epoch
	plt.plot(range(1,num_epochs+1),rms_error, color='black')
	plt.xticks(np.arange(1,num_epochs+1,1))
	plt.title(["Learning Rate = ",'%f'%(learning_rate[rate_num])])	
	
	# Label for x-axis
	plt.xlabel("Epoch Number")
	# Label for y-axis
	plt.ylabel("RMSE")
	# Display graph	
	plt.show()

