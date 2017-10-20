#encoding=utf-8
import sklearn as sk
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import json

from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense

def castingData(dataframe): # 转换数据为一个字典，按年份组织，最后一项是加和
	yearRange = dataframe.index.year
	dataDict = {}
	for year in yearRange:
		singleYear = []
		yearData = dataframe[str(year)]
		monthRange = range(1,13) # 一年12个月
		realMonth = yearData.index.month # 实际存在的月份
		for month in monthRange: # 查找缺失数据，确实默认为-1
			if month in realMonth:
				singleYear.append(float(yearData[str(year)+'-'+str(month)].values.T))
			else:
				singleYear.append(-1)
		yearSum = sum(singleYear) # 加和项
		#yearMean = mean(singleYear)
		singleYear.append(yearSum)
		#singleYear.append(yearMean)

		dataDict[str(year)] = singleYear
	return dataDict


def create_dataset(dataset, look_back=1,forehead=1): # 分割数据，训练集为12个，向前12步预测，参数可调
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-forehead-1):
		a = dataset[i:(i+look_back)]
		b = dataset[(i+look_back):(i+look_back+forehead)]
		dataX.append(a)
		dataY.append(b)
	return np.array(dataX), np.array(dataY)


def newcasting(x,y,look_back,foreNum): # 多步估计

	clf = LinearRegression()
	clf.fit(x,y)
	return clf.predict(x[-1,:]).tolist() # 利用最后一组数据向前预测


# def newcasting(x,y,look_back,foreheadNum):
#
# 	# create and fit Multilayer Perceptron model
# 	model = Sequential()
# 	model.add(Dense(100, input_dim=look_back, activation='relu'))
# 	model.add(Dense(foreheadNum))
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	model.fit(x, y, nb_epoch=100, batch_size=1, verbose=1)
#
# 	# Estimate model performance
# 	trainScore = model.evaluate(x, y, verbose=0)
# 	print('Train Score: ', trainScore)
# 	# testScore = model.evaluate(x, t, verbose=0)
# 	# print('Test Score: ', testScore)
# 	print
# 	return model.predict(x[-1,:].reshape(1,look_back)).tolist()

def evaluate(dataset,look_back,foreNum): # 多步预测直接估计误差

	dataset1 = dataset[:-foreNum]
	x, y = create_dataset(dataset1, look_back, foreNum)


	#clf = MLPRegressor()
	#clf.fit(x,y)
	#ans = clf.predict(x[-1,:])
	ans = newcasting(x, y, look_back, foreNum)
	# ans = rollingForecasting(x,y,testx,12)
	ans = np.array(ans).transpose(1, 0)

	emp = np.empty((dataset1.shape[0], 1))
	emp[:] = np.nan

	error = (dataset1[-foreNum:] - ans[:, 0]) / dataset1[-foreNum:]
	acc = round(1.0-np.mean(np.abs(error)),4)
	print acc
	print error
	np.append(ans,acc)
	MAE = mean_absolute_error(dataset1[-foreNum:], ans)
	MRSE = mean_squared_error(dataset1[-foreNum:], ans)
	print MAE

	return ans.flatten().tolist(),acc
	# print np.sqrt(MRSE)
	# ans = np.vstack((emp, ans))
	# plt.plot(dataset, 'r')
	# plt.plot(ans, 'b')
	# plt.show()

def evaluate2(dataset,look_back,foreNum): # 单步预测估计误差

	dataX, dataY = [], []
	for i in range(len(dataset) - look_back):
		a = dataset[i:(i + look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataX = np.array(dataX)
	dataY = np.array(dataY)

	trainNum = len(dataX)-foreNum
	trainX = dataX[:trainNum]
	trainY = dataY[:trainNum]
	testX = dataX[trainNum:]
	testY = dataY[trainNum:]

	clf = LinearRegression()
	#clf = MLPRegressor()
	clf.fit(trainX,trainY)
	ans = clf.predict(testX)
	accList = (1.0-np.abs(( testY- ans) / testY))*100
	accMean = round(np.mean(np.abs(accList)), 4)

	plt.plot(testY, 'r')
	plt.plot(ans, 'b')
	plt.show()

	return ans.tolist(),accList.tolist(),accMean


if __name__ == "__main__":

	df = pd.read_csv('./data/true_data.csv', encoding='utf-8', index_col='date',skip_footer=0)
	d = df.to_dict()
	#print d['x']
	df.index = pd.to_datetime(df.index)
	dataDcit = castingData(df)
	#print dataDcit
	dataset1 = df['x'].values
	dataset1 = dataset1.astype('float32')
	evalans,accList,accMean = evaluate2(dataset1,12,12)
	print evalans
	print accList
	print accMean







