import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%% matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def airlineForecast(trainingDataFileName, validationDataFileName):

    train_data = getTrainedDataset(trainingDataFileName)
    valid_data = getValidDataset(validationDataFileName)
    final_data = getFinalBookingPerDepartureDate(train_data)
    historical_data = pd.merge(train_data, final_data, how='left', on="departure_date")
    getHistoricalAdditiveDemand(historical_data, valid_data)
    getHistoricalMultiplicativeDemand(historical_data, valid_data)
    getWeeklyAdditiveDemand(historical_data, valid_data)
    getWeeklyMultiplicativeDemand(historical_data, valid_data)
    getWeekVsWeekendAdditiveDemand(historical_data, valid_data)
    getWeekVsWeekendMultiplicativeDemand(historical_data, valid_data)
    getRollingMeanAdditiveDemand(historical_data, valid_data)
    getRollingMeanMultiplicativeDemand(historical_data, valid_data)
    
def getTrainedDataset(trainingDataFileName):
    # import data
    train_data = pd.read_csv("../data/"+trainingDataFileName, sep = ',' , header= 0 )
    # convert the date into datetime format
    train_data["departure_date"] = pd.to_datetime(train_data["departure_date"])
    train_data["booking_date"] = pd.to_datetime(train_data["booking_date"])
    # add new column "days prior" to evaluate forecast data
    train_data["days_prior"] = (train_data["departure_date"] - train_data["booking_date"]).dt.days
    return train_data

def getValidDataset(validationDataFileName):
    # import data 
    valid_data = pd.read_csv("../data/"+validationDataFileName, sep = ',' , header= 0 )
    # convert the date into datetime format
    valid_data["departure_date"] = pd.to_datetime(valid_data["departure_date"])
    valid_data["booking_date"] = pd.to_datetime(valid_data["booking_date"])
    # add days_prior column to enable join with forecasted data
    valid_data["days_prior"] = (valid_data["departure_date"] - valid_data["booking_date"]).dt.days
    valid_data["day_of_week"] = (valid_data["booking_date"]).dt.dayofweek
    return valid_data[["departure_date","days_prior", "cum_bookings", "final_demand", "naive_forecast", "day_of_week"]]

def getHistoricalAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data = historical_data[["days_prior", "net_book"]]
    historical_data = historical_data["net_book"].groupby(historical_data["days_prior"]).mean().reset_index()
    additive_data = pd.merge(valid_data, historical_data, how='left', on="days_prior")
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"]+ additive_data["net_book"]
    print("MASE - Additive Model " + str(calculateMASE(additive_data)))
    
def getHistoricalMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"]/historical_data["final_data"]
    historical_data = historical_data[["days_prior", "multiplicative_rate"]]
    historical_data = historical_data["multiplicative_rate"].groupby(historical_data["days_prior"]).mean().reset_index()
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on="days_prior")
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"]/multiplicative_data["multiplicative_rate"]
    print("MASE - Multiplicative Model " + str(calculateMASE(multiplicative_data)))

def getWeeklyAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data = historical_data[["days_prior", "day_of_week", "net_book"]]
    historical_data = historical_data.groupby(["days_prior","day_of_week"]).mean().reset_index()
    additive_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "day_of_week"])
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"]+ additive_data["net_book"]
    print("MASE - Weekly Additive Model " + str(calculateMASE(additive_data)))

def getWeeklyMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"]/historical_data["final_data"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data = historical_data[["days_prior", "day_of_week", "multiplicative_rate"]]
    historical_data = historical_data.groupby(["days_prior","day_of_week"]).mean().reset_index()
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "day_of_week"])
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"]/multiplicative_data["multiplicative_rate"]
    print("MASE - Weekly Multiplicative Model " + str(calculateMASE(multiplicative_data)))

def getWeekVsWeekendAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data["is_weekend"] = historical_data['day_of_week'].apply(lambda x: 'True' if x in [0,6] else 'False')
    historical_data = historical_data[["days_prior", "is_weekend", "net_book"]]
    historical_data = historical_data.groupby(["days_prior","is_weekend"]).mean().reset_index()
    valid_data["is_weekend"] = valid_data['day_of_week'].apply(lambda x: 'True' if x in [0,6] else 'False')
    additive_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "is_weekend"])
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"]+ additive_data["net_book"]
    print("MASE - Week vs Weekend  Additive Model " + str(calculateMASE(additive_data)))

def getWeekVsWeekendMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"]/historical_data["final_data"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data["is_weekend"] = historical_data['day_of_week'].apply(lambda x: 'True' if x in [0,6] else 'False')
    historical_data = historical_data[["days_prior", "is_weekend", "multiplicative_rate"]]
    historical_data = historical_data.groupby(["days_prior","is_weekend"]).mean().reset_index()
    valid_data["is_weekend"] = valid_data['day_of_week'].apply(lambda x: 'True' if x in [0,6] else 'False')
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "is_weekend"])
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"]/multiplicative_data["multiplicative_rate"]
    print("MASE - Week vs Weekend Multiplicative Model " + str(calculateMASE(multiplicative_data)))

def getRollingMeanAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data = historical_data[["days_prior", "net_book"]]
    test1 = historical_data.groupby('days_prior')['net_book'].apply(lambda x:x.rolling(center=True,window=3).mean())
    historical_data = test1.groupby(historical_data["days_prior"]).mean().reset_index()
    additive_data = pd.merge(valid_data, historical_data, how='left', on="days_prior")
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"]+ additive_data["net_book"]
    print("MASE - Rolling Mean Additive Model " + str(calculateMASE(additive_data)))

def getRollingMeanMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"]/historical_data["final_data"]
    historical_data = historical_data[["days_prior", "multiplicative_rate"]]
    test1 = historical_data.groupby('days_prior')['multiplicative_rate'].apply(lambda x:x.rolling(center=True,window=2).mean())
    historical_data = test1.groupby(historical_data["days_prior"]).mean().reset_index()
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior"])
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"]/multiplicative_data["multiplicative_rate"]
    print("MASE - Rolling Mean Multiplicative Model " + str(calculateMASE(multiplicative_data)))

def getFinalBookingPerDepartureDate(final_data):
    final_data = final_data.loc[final_data['days_prior'] == 0]
    final_data["final_data"] = final_data["cum_bookings"]
    return final_data[["departure_date", "final_data"]]

def calculateMASE(model_data):
    model_data["naive_error"] = (model_data["final_demand"] - model_data["naive_forecast"]).abs()
    model_data["forecast_error"] = (model_data["final_demand"] - model_data["final_forecast"]).abs()
    naive_error = model_data["naive_error"].sum(axis=0)
    model_error = model_data["forecast_error"].sum(axis=0)
    # calculate MASE 
    mase = (model_error/naive_error)
    return mase

def main():
    airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')


main()


























