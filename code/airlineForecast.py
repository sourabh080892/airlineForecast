import os
import pandas as pd
import numpy as np


def airlineForecast(trainingDataFileName, validationDataFileName):

    # import both the data into program
    train_data = pd.read_csv(trainingDataFileName, sep = ',' , header= 0 )
    valid_data = pd.read_csv(validationDataFileName, sep = ',' , header= 0 )

    # convert the date into calculation possible format
    train_data["departure_date"] = pd.to_datetime(train_data["departure_date"])
    train_data["booking_date"] = pd.to_datetime(train_data["booking_date"])
    valid_data["departure_date"] = pd.to_datetime(valid_data["departure_date"])
    valid_data["booking_date"] = pd.to_datetime(valid_data["booking_date"])
    valid_data["add_forecast"] = 0
    valid_data["mul_forecast"] = 0

    # changes in train_data data frame
    # add new column "days prior"
    train_data["days_prior"] = (train_data["departure_date"] - train_data["booking_date"]).dt.days

    # find net booking from cumulative bookings
    train_data["net_book"] = train_data.groupby(["departure_date"])["cum_bookings"].diff()

    # Iterate over rows of Valid data
    for index, row in valid_data.iterrows():
        days_prior = (row["departure_date"] - row["booking_date"])
        days_prior = days_prior.days
        if days_prior == 0:
            continue
        else:
            # Additive model
            forecast_add = additiveBooking(train_data, days_prior)
            valid_data.loc[index, "add_forecast"] = forecast_add + row["cum_bookings"]

            # multiplicative model
            forecast_mul = multiplicativeBooking(train_data, days_prior)
            valid_data.loc[index, "mul_forecast"] = row["cum_bookings"] / forecast_mul

    # calculate the MASE value
    mase_add = calculateMASE(valid_data, "add_forecast")
    mase_mul = calculateMASE(valid_data, "mul_forecast")
    return mase_add


def additiveBooking(train_data, days_prior):
    # fetch the rows where days prior is less than or equal to 8
    prior_train_data = train_data.loc[train_data["days_prior"] < days_prior]
    prior_cum_book = prior_train_data.groupby(["departure_date"])["net_book"].sum().reset_index()

    return prior_cum_book["net_book"].mean()


def multiplicativeBooking(train_data, days_prior):
    # fetch the rows where days prior is less than or equal to 8
    prior_train_data = train_data.loc[train_data["days_prior"] == days_prior]
    prior_total_book = train_data['cum_bookings'].groupby(train_data['departure_date']).max().reset_index()
    prior_total_book.columns = ['departure_date', 'total_book']
    prior_train_data = pd.merge(prior_train_data, prior_total_book, on="departure_date")
    prior_train_data.drop(["booking_date", "net_book"], 1)      # not working please check
    prior_train_data["book_rate"] = prior_train_data["cum_bookings"] / prior_train_data["total_book"]

    return prior_train_data["book_rate"].mean()


def calculateMASE(valid_data, forecast_column):
    naive_error = abs(valid_data["final_demand"] - valid_data["naive_forecast"]).sum()

    # replace all values, 0 by NaN, so that they will not be considered as a part of calculation
    valid_data[forecast_column] = valid_data[forecast_column].replace(0, np.nan)
    model_error = abs(valid_data["final_demand"] - valid_data[forecast_column]).sum()

    # calculate MASE error
    mase = (model_error / naive_error) * 100
    return mase


def main():
    os.chdir("E:/SeattleU Education//1st Quarter, Fall18/IS-5201 Programming for Business Analytics/Group Project/data")
    airlineForecast('airline_booking_trainingData.csv', 'airline_booking_validationData.csv')


main()


























