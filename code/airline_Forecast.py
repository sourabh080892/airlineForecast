import os
import pandas as pd
import numpy as np


def airlineForecast(trainingDataFileName, validationDataFileName):
    train_data = getTrainedDataset(trainingDataFileName)
    valid_data = getValidDataset(validationDataFileName)
    final_data = getFinalBookingPerDepartureDate(train_data)
    historical_data = pd.merge(train_data, final_data, how='left', on="departure_date")

    additive_data = getHistoricalAdditiveDemand(historical_data, valid_data)
    print("MASE - Additive Model " + str(calculateMASE(additive_data)))

    multiplicative_data = getHistoricalMultiplicativeDemand(historical_data, valid_data)
    print("MASE - Multiplicative Model " + str(calculateMASE(multiplicative_data)))

    dow_book_additive_data = getWeeklyAdditiveDemand(historical_data, valid_data)
    print("MASE - Booking DOW Additive Model " + str(calculateMASE(dow_book_additive_data)))

    dow_book_multiplicative_data = getWeeklyMultiplicativeDemand(historical_data, valid_data)
    print("MASE - Booking DOW Multiplicative Model " + str(calculateMASE(dow_book_multiplicative_data)))

    # divide validation data on the basis of weekday(wday) and weekend(wend)
    valid_data_wday = valid_data[valid_data.depart_dow_no <= 3]
    valid_data_wend = valid_data[valid_data.depart_dow_no >= 4]

    # run additive model on weekdays and multiplicative on weekend
    additive_data_wday = getHistoricalAdditiveDemand(historical_data, valid_data_wday)
    multiplicative_data_wend = getHistoricalMultiplicativeDemand(historical_data, valid_data_wend)

    additive_data_wday = additive_data_wday[["final_demand", "naive_forecast", "final_forecast"]]
    multiplicative_data_wend = multiplicative_data_wend[["final_demand", "naive_forecast", "final_forecast"]]
    model_wday_wend = additive_data_wday.append(multiplicative_data_wend, ignore_index=True)
    print("MASE - Departure DOW Additive + Multiplicative Model " + str(calculateMASE(model_wday_wend)))

    #Average booking smoothing model
    train_daily_booking_3 = train_depDOW(train_data, valid_data)
    print("MASE - Average booking smoothing(3) model " + str(calculateMASE(train_daily_booking_3)))

def getTrainedDataset(trainingDataFileName):
    # import data
    train_data = pd.read_csv("../data/" + trainingDataFileName, sep=',', header=0)
    # convert the date into datetime format
    train_data["departure_date"] = pd.to_datetime(train_data["departure_date"])
    train_data["booking_date"] = pd.to_datetime(train_data["booking_date"])
    # add new column "days prior" to evaluate forecast data
    train_data["days_prior"] = (train_data["departure_date"] - train_data["booking_date"]).dt.days
    train_data["departure_dow"] = train_data["departure_date"].dt.weekday_name
    train_data["net_book"] = train_data.groupby(["departure_date"])["cum_bookings"].diff()
    train_data["net_book"] = train_data["net_book"].replace(np.nan, 0)
    train_data["smooth_book_avg_3"] = (train_data['net_book'] + train_data['net_book'].shift(1) + train_data['net_book'].shift(-1)) / 3
    # Set first row of new departure date set to 0
    train_data["smooth_book_avg_3"] = np.where(train_data['days_prior'].shift(1) == 0, 0, train_data['smooth_book_avg_3'])
    return train_data


def getValidDataset(validationDataFileName):
    # import data
    valid_data = pd.read_csv("../data/" + validationDataFileName, sep=',', header=0)
    # convert the date into datetime format
    valid_data["departure_date"] = pd.to_datetime(valid_data["departure_date"])
    valid_data["booking_date"] = pd.to_datetime(valid_data["booking_date"])
    # add days_prior column to enable join with forecasted data
    valid_data["days_prior"] = (valid_data["departure_date"] - valid_data["booking_date"]).dt.days
    valid_data["departure_dow"] = valid_data["departure_date"].dt.weekday_name
    # add new column for day of week : departure date
    valid_data["day_of_week"] = (valid_data["booking_date"]).dt.dayofweek
    valid_data["depart_dow_no"] = (valid_data["departure_date"]).dt.dayofweek
    return valid_data[["departure_date", "days_prior", "cum_bookings", "final_demand", "naive_forecast", "day_of_week",
                       "departure_dow", "depart_dow_no"]]


def getHistoricalAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data = historical_data[["days_prior", "net_book"]]
    historical_data = historical_data["net_book"].groupby(historical_data["days_prior"]).mean().reset_index()
    additive_data = pd.merge(valid_data, historical_data, how='left', on="days_prior")
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"] + additive_data["net_book"]
    return additive_data


def getHistoricalMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"] / historical_data["final_data"]
    historical_data = historical_data[["days_prior", "multiplicative_rate"]]
    historical_data = historical_data["multiplicative_rate"].groupby(historical_data["days_prior"]).mean().reset_index()
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on="days_prior")
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"] / multiplicative_data[
        "multiplicative_rate"]
    return multiplicative_data


def getWeeklyAdditiveDemand(historical_data, valid_data):
    historical_data["net_book"] = historical_data["final_data"] - historical_data["cum_bookings"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data = historical_data[["days_prior", "day_of_week", "net_book"]]
    historical_data = historical_data.groupby(["days_prior", "day_of_week"]).mean().reset_index()
    additive_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "day_of_week"])
    additive_data = additive_data[additive_data.days_prior != 0]
    additive_data["final_forecast"] = additive_data["cum_bookings"] + additive_data["net_book"]
    return additive_data


def getWeeklyMultiplicativeDemand(historical_data, valid_data):
    historical_data["multiplicative_rate"] = historical_data["cum_bookings"] / historical_data["final_data"]
    historical_data["day_of_week"] = (historical_data["booking_date"]).dt.dayofweek
    historical_data = historical_data[["days_prior", "day_of_week", "multiplicative_rate"]]
    historical_data = historical_data.groupby(["days_prior", "day_of_week"]).mean().reset_index()
    multiplicative_data = pd.merge(valid_data, historical_data, how='left', on=["days_prior", "day_of_week"])
    multiplicative_data = multiplicative_data[multiplicative_data.days_prior != 0]
    multiplicative_data["final_forecast"] = multiplicative_data["cum_bookings"] / multiplicative_data[
        "multiplicative_rate"]
    return multiplicative_data


def getFinalBookingPerDepartureDate(final_data):
    final_data = final_data.loc[final_data['days_prior'] == 0]
    final_data["final_data"] = final_data["cum_bookings"]
    return final_data[["departure_date", "final_data"]]


def train_depDOW(train, valid_data):
    train_dow = pd.DataFrame(index=range(0, 60),
                          columns=['All', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Setup dataframe to columns of interest only
    train = train[["days_prior", "smooth_book_avg_3", "departure_dow"]]
    train_dow['All'] = train.groupby("days_prior").mean().astype(float)

    train_dow['Monday'] = (train[train['departure_dow'] == 'Monday'].groupby("days_prior")).mean().astype(float)
    train_dow['Tuesday'] = (train[train['departure_dow'] == 'Tuesday'].groupby("days_prior")).mean().astype(float)
    train_dow['Wednesday'] = (train[train['departure_dow'] == 'Wednesday'].groupby("days_prior")).mean().astype(float)
    train_dow['Thursday'] = (train[train['departure_dow'] == 'Thursday'].groupby("days_prior")).mean().astype(float)
    train_dow['Friday'] = (train[train['departure_dow'] == 'Friday'].groupby("days_prior")).mean().astype(float)
    train_dow['Saturday'] = (train[train['departure_dow'] == 'Saturday'].groupby("days_prior")).mean().astype(float)
    train_dow['Sunday'] = (train[train['departure_dow'] == 'Sunday'].groupby("days_prior")).mean().astype(float)

    valid_avg3_data = valid_data.copy()
    valid_avg3_data['final_forecast'] = valid_avg3_data['cum_bookings'] + \
                                        valid_avg3_data[valid_avg3_data.days_prior > 0][
                                              ['days_prior', 'departure_dow']].apply(lambda row: sum(
                                              [train_dow.loc[i, row[1]] for i in range(row[0] - 1, -1, -1)]), axis=1)

    return valid_avg3_data


def calculateMASE(model_data):
    model_data["naive_error"] = (model_data["final_demand"] - model_data["naive_forecast"]).abs()
    model_data["forecast_error"] = (model_data["final_demand"] - model_data["final_forecast"]).abs()
    naive_error = model_data["naive_error"].sum(axis=0)
    model_error = model_data["forecast_error"].sum(axis=0)
    # calculate MASE
    mase = (model_error / naive_error)
    return mase


def main():
    os.chdir("E:/SeattleU Education//1st Quarter, Fall18/IS-5201 Programming for Business Analytics/Group Project/data")
    airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')


main()
