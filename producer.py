import pandas as pd
import time
import shutil


def cast_date(dataPath):
	df = pd.read_csv(dataPath,sep = ",",header=0)
	df['Date']=df.Date.str.slice(0,19)
	splittedDate=df.Date.str.split()
	df['Date']=[dateTime[0] for dateTime in splittedDate]
	df['Time']=[dateTime[1] for dateTime in splittedDate]

	splittedDate=df.Date.str.split('/')
	df['Month']=[int(date[0]) for date in splittedDate]
	df['Day']=[int(date[1]) for date in splittedDate]

	splittedTime=df.Time.str.split(':')
	df['Hour']=[int(Time[0]) for Time in splittedTime]
	df['Minute']=[int(Time[1]) for Time in splittedTime]

	df = df.drop(columns=['Date','Time','Unnamed: 0'])
	df.info()
	df.to_csv('Casted_Date_Crimes_2012-2017.csv')

def produce():
	cast_date("Chicago_Crimes_2012_to_2017.csv")
	url = 'Casted_Date_Crimes_2012-2017.csv'
	df = pd.read_csv(url,sep = ",",header=0)
	df = df.drop(columns=['Unnamed: 0'])
	for i in range(1,13):
		name = r'batches/batch_month{m}.csv'.format(m=i)
		print(name)
		df.loc[(df.Month == i)].to_csv(name, index = False, header=True)
		#shutil.copy(name,"tmp/")
		time.sleep(150)

produce()
