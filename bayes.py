import csv, sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp

#Function defined to read csv file and create header line, target and independent variables
def get_data(filename):
	dataset = []
	header = []
	target = []
	rec_count = 0
	header_line = 1
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			if header_line == 0:
				dataset.append([])
				for i in range(1,len(row)):
					#separate target and independent variables
					if i ==1:
						target.append(int(row[i]))
					else:
						dataset[rec_count].append(int(row[i]))
				rec_count+=1
			else:
				header.append (row)
 				header_line = 0			
	return header,target,dataset

#Create database and copy data to it from the csv file
def copytodb(filename):
	con = sqlite3.connect("ds2.db")
	cur = con.cursor()
	#create table
	#cur.execute("DROP TABLE IF EXISTS `pilotdata`;")
	cur.execute("CREATE TABLE IF NOT EXISTS pilotdata (obs, testres, var2, var3,var4,var5,var6 );") 
	with open(filename,'rb') as f:
	    # csv.DictReader uses first line in file for column headings by default
	    dr = csv.DictReader(f)
	    to_db = [(i['Obs'], i['TestRes/Var1'],i['Var2'],i['Var3'],i['Var4'],i['Var5'],i['Var6']) for i in dr]
	#copy observations to the database
	cur.executemany("INSERT INTO pilotdata (obs, testres, var2, var3,var4,var5,var6) VALUES (?,?,?,?,?,?,?);", to_db)
	#confirm changes to the database
	con.commit()
	con.close()

def readfromdb():
	con = sqlite3.connect("ds2.db")
	cur = con.cursor()
	cur.execute("SELECT count(*) from sqlite_master where type='table' and name='pilotdata';")
	if cur.fetchone()[0]==1:
		#Read header row and copy to array
		cur.execute('SELECT sql FROM sqlite_master WHERE tbl_name = \'pilotdata\' and type = \'table\';')
		header = cur.fetchall()
		#Read target column and copy to array
		cur.execute('SELECT testRes FROM pilotdata;') 
		target1 = cur.fetchall()
		#Read independent variables and copy to array
		cur.execute('SELECT Var2, Var3, Var4, Var5, Var6 FROM pilotdata;') 
		dataset = cur.fetchall()
		con.close()
		database = []
		target = []
		#convert unicode data to integers
		for i in range(len(dataset)):
			database.append([])
			target.append(int(target1[i][0]))
			for j in range(len(dataset[i])):
				database[i].append(int(dataset[i][j]))
		return header, target, database
	else:
		print "Table does not exist. Please create table in database."
		exit()

#function to delete table at end 
def deletetable(tablename):
	con = sqlite3.connect("ds2.db")
	cur = con.cursor()
	query = "DROP TABLE " + tablename + ";"
	cur.execute(query) 
	con.commit()
	con.close()
#calculate number of unique values for each variable
def unique(list1):
    x = np.array(list1)
    return np.unique(x)
    
#get probabilties for each variable outcome using the count for target values of 0 and 1
def get_prob(unique, count_zero, count_one,indep_var, target):
	prob_zero = []
	prob_one = []
	for i in range(len(unique)):
		prob_zero.append([])
		prob_one.append([])
		for j in range(len(unique[i])):
			prob_zero[i].append(float(count_zero[i][str(unique[i][j])]/float(sum(target))))
			prob_one[i].append(float(count_one[i][str(unique[i][j])]/float(len(target)-sum(target))))
		print prob_zero[i],prob_one[i]
	return prob_zero, prob_one

#calculate probability using the input data for validation
def calc_tar_prob(prob_zero,prob_one, indep_var):
	prob_array=[]
	for i in range(len(indep_var)):
		prod0 =1.0
		prod1 = 1.0
		for j in range(0,len(indep_var[i])):
			prod0 *= prob_zero[j][indep_var[i][j]]				
			prod1 *= prob_one[j][indep_var[i][j]]
		if (prod0+prod1) == 0:
			prob_array.append(0)	
		else:
			prob_array.append(float(prod1/float(prod0+prod1)))
	print prob_array	
	return prob_array
	
#implementation of the naive bayes algorithm
def naive_bayes(header, target, indep_var):
	prob_tar_zero = []
	prob_tar_one = []
	unique_vals = []
	for i in range(len(indep_var[0])):
		#dictionaries to count number of occurences of each outcome for each variable
		dict_zero = {} #to store count when target = 0
		dict_one = {}  #to store count when target = 1
		unique_vals.append(list(unique(np.array(indep_var)[:,i])))
		for j in range(len(unique_vals[i])):
			#store unique keys in the dictionaries
			dict_zero.update({str(unique_vals[i][j]):0})
			dict_one.update({str(unique_vals[i][j]):0})
		for j in range(len(indep_var)):
			if target[j] == 0: 
				#get count for each key in the dictionaries
				dict_zero[str(indep_var[j][i])]=dict_zero[str(indep_var[j][i])]+1
			else:
				dict_one[str(indep_var[j][i])]=dict_one[str(indep_var[j][i])]+1
		prob_tar_zero.append(dict_zero)
		prob_tar_one.append(dict_one)

	#function to create count into probabilties
	prob_zero,prob_one = get_prob(unique_vals,prob_tar_zero,prob_tar_one, indep_var, target)
	
	prob_array = calc_tar_prob(prob_zero,prob_one,indep_var)	
	return prob_array
   
def calc_roc(prob_array,indep_var, target):
	roc_table=[]
	#create table containing probabilities and target values
	for i in range(len(indep_var)):
		roc_table.append([])
		roc_table[i].append(prob_array[i])
		roc_table[i].append(target[i])
	#sort the table in ascending order by probabilities
	roc_table = sorted(roc_table, key = lambda x:(x.__getitem__(0)))
	return roc_table

def calc_positive_rates(roc_table, indep_var,target):	
	threshold = 0.0
	#arrays to store true positive and false positive rates	
	tpr= []
	fpr = []
	
	for i in range(len(roc_table)):
		#arrays to store count of true positives and false positives		
		true_pos = []
		false_pos = []
		threshold = roc_table[i][0]
		for j in range(0,len(roc_table)):
			if roc_table[j][0] >= threshold:
				if roc_table[j][1]==1:
					true_pos.append(1)
				else:
					false_pos.append(1)
		#true positive rate = number of true positives / actual positives
		tpr.append(float(sum(true_pos))/float(sum(target)))
		#false positive rate = number of false positives / actual negatives
		fpr.append(float(sum(false_pos))/float(len(target)-sum(target)))
	print tpr
	print fpr
	return tpr, fpr

def print_roc(tpr, fpr):
	#print the ROC curve using the true positive rate and false positive rate
	plt.plot(fpr, tpr,color='blue')
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Curve of True Positive Rate vs False Positive Rate")
	
	plt.gca().set_xlim([-0.01,1.01])
	plt.gca().set_ylim([-0.01,1.01])
	plt.plot(plt.gca().get_xlim(),plt.gca().get_ylim(),color = 'green')
	plt.show()

print "Select from the following options"
print "1. Read from file"
print "2. Read from database"
opt = raw_input()
if opt == "1":	
	header, target, indep_var = get_data("Flying_Fitness.csv")
	prob_array = naive_bayes(header, target, indep_var)
	roc_table = calc_roc(prob_array,indep_var, target)
	tpr, fpr = calc_positive_rates(roc_table, indep_var, target)
	print_roc(tpr,fpr)
else:
	print "Do you want to write to database? (y/n)"
	opt = raw_input()
	if opt == 'y':
		copytodb("Flying_Fitness.csv")
	header, target, indep_var = readfromdb()
	prob_array = naive_bayes(header, target, indep_var)
	roc_table = calc_roc(prob_array,indep_var, target)
	tpr, fpr = calc_positive_rates(roc_table, indep_var, target)
	print_roc(tpr,fpr)
	print "Do you want to delete the database? (y/n)"
	opt = raw_input()
	if opt == 'y':
		deletetable("pilotdata")

