from Classifier import load
import numpy as np
from sklearn.preprocessing import scale,normalize
import math
def cleanFeatures(X_train):

	# remove ID's
	X_train = X_train[:,1:]

	# create goal/day ratio
	dollarPerDay = np.divide(X_train[:,0],X_train[:,1])
	X_train = np.column_stack((X_train, dollarPerDay))
	# standardize goal
	scale(X_train[:,0],copy = False)
	# standardize goal/day ratio
	scale(X_train[:,-1],copy = False)
	X_train[:,-1] *= 2	
	
	# time data- backers. Split number of backers into 3 time sets
	backers = [[0,0,0,0]]
	for row in X_train:
		day12 = row[5+3]
		day34 = row[3*5+3]-day12
		day57 = row[6*5+3]-day12-day34
		day810 = row[9*5+3]-day12-day34-day57
		backers = np.concatenate((backers,[[day12,day34,day57,day810]]),axis = 0)
	X_train = np.concatenate((X_train,backers[1:,:]),axis = 1)
	scale(X_train[:,-4:],copy = False)
	
	# time data- tweets. split number of tweets into 3 time sets
	'''
	This section was used to compute the time set data for tweets. It was only used in the best performing classifier submitted on Kaggle
	'''
	#tweets = [[0,0,0]]
	#for row in X_train:
	#	day13 = row[14]
	#	day46 = row[29]-day13
	#	day710 = row[9*5+4]-day46-day13
	#	tweets = np.concatenate((tweets,[[day13,day46,day710]]),axis = 0)
	#X_train = np.concatenate((X_train,tweets[1:,:]),axis = 1)
	#scale(X_train[:,-3:],copy = False)	
	
	# time data- retweets. split number of retweets into 3 time sets
	retweets = [[0,0,0]]
	for row in X_train:
		day13 = row[2*5+6]
		day46 = row[5*5+6]-day13
		day710 = row[9*5+6]-day46-day13
		retweets = np.concatenate((retweets,[[day13,day46,day710]]),axis = 0)
	X_train = np.concatenate((X_train,retweets[1:,:]),axis = 1)
	scale(X_train[:,-3:],copy = False)	
	
	# has replies
	replies = [[0]]
	for row in X_train:
		if row[9*5+5] >= 1:
			replies = np.concatenate((replies,[[1]]),axis = 0)
		else:
			replies = np.concatenate((replies,[[0]]),axis = 0)
	X_train = np.concatenate((X_train,replies[1:,:]),axis = 1)	
	return X_train

def getFeatures(X_train,a, c):
	# linear regression vs time for number of tweets
	'''
	This section calculates linear regression for number of tweets over a ten day period. This function was not used in the final classifier, but was used in the second-best performing classifier and in almost all other tests.
	'''
	x = [i for i in range(1,11)]
	tweetReg = [[0]]
	normalize(X_train[:,4:50:5],copy = False)
	for row in X_train: 
		y = [row[i*5+4] for i in range(0,10)]
		reg = np.polyfit(x, y, 1, full = False)
		r = reg[0]
		tweetReg = np.concatenate((tweetReg,[[r]]),axis = 0)
	X_train = np.concatenate((X_train,tweetReg[1:,:]),axis = 1)		
	normalize(X_train[:,-1],copy = False)
	
	#stagnated? based on trend in tweets per day
	'''
	This section calculates the stagnated feature. Used in the second-best performing classifier.
	'''
	#stagnated = []
	#for row in X_train:
	#	x = [i for i in range(1,11)]
	#	y = [row[i*5+4] for i in range(0,10)]
	#	regr = np.polyfit(x,y,1,full = False)
	#	if math.fabs(regr[0]) < 0.0001:
	#		stagnated.append(1)
	#	else:
	#		stagnated.append(0)
	#X_train = np.column_stack((X_train,np.array(stagnated)))

	# fit exponential curve for goal
	flag = True
	onTrack = []
	for row in X_train:
		x = 10/row[1]
		y = [row[i*5+2] for i in range(0,10)]
		target = np.exp([c*(v-x) if (c*(v-x)) < 15 else 15 for v in y])
		target = [i*a for i in target]
		onTrack.append(target[9])
	X_train = np.column_stack((X_train,onTrack))

	## linear regression vs time for number of backers
	'''
	Calculate linear regression for number of backers. Used in initial tests, but not for any of the final classifiers.
	'''
	#x = [i for i in range(1,11)]
	#backersReg = [[0,0]]
	#normalize(X_train[:,3:5:49],copy = False)
	#for row in X_train: 
	#	y = [row[i*5+3] for i in range(0,10)]
	#	reg = np.polyfit(x, y, 1, full = False)
	#	backersReg = np.concatenate((backersReg,[reg]),axis = 0)
	#X_train = np.concatenate((X_train,backersReg[1:,:]),axis = 1)
	#scale(X_train[:,-2:],copy = False)
	
	# delete superfluous columns
	X_train = np.delete(X_train, [i*5+4 for i in range(0,10)],1) # n tweets
	X_train = np.delete(X_train, [i*4+2 for i in range(0,10)],1) # pledge
	X_train = np.delete(X_train, [i*3+2 for i in range(0,10)],1) # n backers
	X_train = np.delete(X_train, [i*2+2 for i in range(0,10)],1) # n replies
	X_train = np.delete(X_train, [i+2 for i in range (0,10)],1) # n retweets
	scale(X_train[:,1],copy = False)
	return X_train

