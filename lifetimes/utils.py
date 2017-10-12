"""Utility functions for lifetime calculations."""

import datetime as dt
import numpy as np
import pandas as pd
from scipy.optimize import fmin

pd.options.mode.chained_assignment = None

__all__ = ['calibration_and_holdout_data',
           'find_first_transactions',
           'summary_data_from_transaction_data',
           'calculate_alive_path',
           'customer_lifetime_value']


def coalesce(*args):
    return next(s for s in args if s is not None)


def calibration_and_holdout_data(transactions, customer_id_col, datetime_col,
                                 calibration_period_end,
                                 calibration_period_start=None,
                                 observation_period_end=None,
                                 freq='D', datetime_format=None,
                                 monetary_value_col=None):
    """Create a summary of calibration and holdout data from transaction data.

    This function creates a summary of each customer over a calibration and
    holdout period (training and testing, respectively). It accepts transaction
    data, and returns a Dataframe of sufficient statistics.

    Parameters:
        transactions: a Pandas DataFrame of at least two cols.
        customer_id_col: the column in transactions that denotes the
            customer_id
        datetime_col: the column in transactions that denotes the datetime the
            purchase was made.
        calibration_period_end: a period to limit the calibration to,
            inclusive.
        observation_period_end: a string or datetime to denote the final date
            of the study. Events after this date are truncated, inclusive.
        freq: Default 'D' for days. Other examples: 'W' for weekly.
        datetime_format: a string that represents the timestamp format. Useful
            if Pandas can't understand the provided format.
        monetary_value_col: the column in transactions that denotes the
            monetary value of the transaction. Optional, only needed for
            customer lifetime value estimation models.

    Returns:
        A dataframe with columns frequency_cal, recency_cal, T_cal,
        frequency_holdout, and duration_holdout.
        If monetary_value_col isn't None, the dataframe will also have the
        columns monetary_value_cal and monetary_value_holdout.
    """
    def to_period(d):
        return d.to_period(freq)

    # make a copy of only relevant information
    transaction_cols = [customer_id_col, datetime_col]
    if monetary_value_col:
        transaction_cols.append(monetary_value_col)
    transactions = transactions[transaction_cols].copy()

    # transform dates to DateTimes
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col],
                                                format=datetime_format)
    # optional arguments
    if observation_period_end is None:
        observation_period_end = transactions[datetime_col].max()
    else:
        observation_period_end = pd.to_datetime(observation_period_end,
                                                format=datetime_format)
    if calibration_period_start is None:
        calibration_period_start = transactions[datetime_col].min()
    else:
        calibration_period_start = pd.to_datetime(calibration_period_start,
                                                  format=datetime_format)
    # this argument is needed
    calibration_period_end = pd.to_datetime(calibration_period_end,
                                            format=datetime_format)

    # create calibration dataset
    calibration_transactions = transactions.loc[(transactions[datetime_col] >=
                                                calibration_period_start) &
                                                (transactions[datetime_col] <=
                                                calibration_period_end), :]

    calibration_summary_data = summary_data_from_transaction_data(
        calibration_transactions,
        customer_id_col,
        datetime_col,
        datetime_format=datetime_format,
        observation_period_end=calibration_period_end,
        freq=freq,
        monetary_value_col=monetary_value_col)
    calibration_summary_data.columns = [c + '_cal' for c in
                                        calibration_summary_data.columns]

    # create holdout dataset
    holdout_transactions = transactions.loc[(observation_period_end >=
                                            transactions[datetime_col]) &
                                            (transactions[datetime_col] >
                                            calibration_period_end), :]

    # convert DatetimeIndex to PeriodIndex
    holdout_transactions[datetime_col] = holdout_transactions[datetime_col] \
        .map(to_period)

    holdout_summary_data = pd.DataFrame()
    holdout_summary_data['frequency_holdout'] = holdout_transactions.groupby(
        customer_id_col, sort=False)[datetime_col].agg('count')

    if monetary_value_col:
        holdout_summary_data['monetary_value_holdout'] = holdout_transactions \
            .groupby(customer_id_col)[monetary_value_col].mean()

    combined_data = calibration_summary_data.join(holdout_summary_data,
                                                  how='left')
    combined_data.fillna(0, inplace=True)
    delta_time = to_period(observation_period_end) - \
        to_period(calibration_period_end)
    combined_data['duration_holdout'] = delta_time

    return combined_data


def find_first_transactions(transactions, customer_id_col, datetime_col,
                            monetary_value_col=None, datetime_format=None,
                            observation_period_end=dt.date.today(), freq='D'):
    """Find first transactions in transaction log data.

    This takes a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    and appends a boolean column named 'first' to the transaction log that
    indicates which transaction is first and which are not (thus repeated
    transactions) for every customer_id.

    Parameters:
        transactions: A Pandas DataFrame.
        customer_id_col: The column in transactions that denotes the
            customer_id.
        datetime_col: The column in transactions that denotes the datetime the
            purchase was made.
        monetary_value_col (optional): The column in transactions that denotes
            the monetary value of the transaction. Only needed for customer
            lifetime value estimation models.
        observation_period_end: A string or datetime to denote the final date
            of the study. Events after this date are truncated.
        datetime_format: A string that represents the timestamp format. Useful
            if Pandas can't understand the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc.
            Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    select_columns = [customer_id_col, datetime_col]

    if monetary_value_col:
        select_columns.append(monetary_value_col)

    transactions = transactions[select_columns].sort_values(
        select_columns).copy()

    # make sure the date column uses datetime objects, and use Pandas'
    # DateTimeIndex.to_period() to convert the column to a PeriodIndex which is
    # useful for time-wise grouping and truncating
    transactions[datetime_col] = pd.to_datetime(transactions[datetime_col],
                                                format=datetime_format)
    transactions = transactions.set_index(datetime_col).to_period(freq)

    transactions = transactions.ix[(transactions.index <=
                                    observation_period_end)].reset_index()

    period_groupby = transactions.groupby([datetime_col, customer_id_col],
                                          sort=False, as_index=False)

    if monetary_value_col:
        # when we have a monetary column, make sure to sum together any values
        # in the same period
        period_transactions = period_groupby.sum()
    else:
        # by calling head() on the groupby object, the datetime_col and
        # customer_id_col columns will be reduced
        period_transactions = period_groupby.head(1)

    # initialize a new column where we will indicate which are the first
    # transactions
    period_transactions['first'] = False
    # find all of the initial transactions and store as an index
    first_transactions = period_transactions.groupby(customer_id_col,
                                                     sort=True,
                                                     as_index=False
                                                     ).head(1).index
    # mark the initial transactions as True
    period_transactions.loc[first_transactions, 'first'] = True
    select_columns.append('first')

    return period_transactions[select_columns]


def summary_data_from_transaction_data(transactions, customer_id_col,
                                       datetime_col, monetary_value_col=None,
                                       datetime_format=None,
                                       observation_period_end=dt.date.today(),
                                       freq='D'):
    """Create summary data from transaction data.

    This transforms a Dataframe of transaction data of the form:
        customer_id, datetime [, monetary_value]
    to a Dataframe of the form:
        customer_id, frequency, recency, T [, monetary_value]

    Parameters:
        transactions: A Pandas DataFrame.
        customer_id_col: The column in transactions that denotes the
            customer_id.
        datetime_col: The column in transactions that denotes the datetime the
            purchase was made.
        monetary_value_col (optional): The columns in the transactions that
            denotes the monetary value of the transaction. Only needed for
            customer lifetime value estimation models.
        observation_period_end: A string or a datetime to denote the final date
            of the study. Events after this date are truncated.
        datetime_format: A string that represents the timestamp format. Useful
            if Pandas can't understand the provided format.
        freq: Default 'D' for days, 'W' for weeks, 'M' for months... etc.
            Full list here:
            http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects
    """
    observation_period_end = pd.to_datetime(observation_period_end,
                                            format=datetime_format) \
                               .to_period(freq)

    # label all of the repeated transactions
    repeated_transactions = find_first_transactions(
        transactions,
        customer_id_col,
        datetime_col,
        monetary_value_col,
        datetime_format,
        observation_period_end,
        freq
    )
    # count all orders by customer.
    customers = repeated_transactions.groupby(customer_id_col,
                                              sort=False)[datetime_col] \
                                     .agg(['min', 'max', 'count'])

    # subtract 1 from count, as we ignore their first order.
    customers['frequency'] = customers['count'] - 1

    customers['T'] = (observation_period_end - customers['min'])
    customers['recency'] = (customers['max'] - customers['min'])

    summary_columns = ['frequency', 'recency', 'T']

    if monetary_value_col:
        # create an index of all the first purchases
        first_purchases = repeated_transactions[repeated_transactions['first']] \
            .index
        # by setting the monetary_value cells of all the first purchases to
        # NaN, those values will be excluded from the mean value calculation
        repeated_transactions.loc[first_purchases, monetary_value_col] = np.nan
        customers['monetary_value'] = repeated_transactions \
            .groupby(customer_id_col)[monetary_value_col].mean().fillna(0)
        summary_columns.append('monetary_value')

    return customers[summary_columns].astype(float)


def calculate_alive_path(model, transactions, datetime_col, t, freq='D'):
    """Calculate customer's probability of being alive as a function of time.

    Parameters:
        model: A fitted lifetimes model.
        transactions: A Pandas DataFrame containing the transactions history
            of the customer_id.
        datetime_col: The column in the transactions that denotes the
            datetime the purchase was made.
        t: The number of time units since the birth for which we want to draw
            the p_alive.
        freq: A Pandas timeseries frequency. Default 'D' for days. For example,
            use 'W' for weekly.

    Returns:
        A Pandas Series containing the p_alive as a function of T (age of the
            customer).
    """
    customer_history = transactions[[datetime_col]].copy()
    customer_history[datetime_col] = pd.to_datetime(
        customer_history[datetime_col])
    customer_history = customer_history.set_index(datetime_col)
    # add transactions column
    customer_history['transactions'] = 1
    purchase_history = customer_history.resample(freq, how='sum') \
        .fillna(0)['transactions'].values
    extra_columns = t - len(purchase_history)
    customer_history = pd.DataFrame(np.append(purchase_history,
                                              [0] * extra_columns),
                                    columns=['transactions'])
    # add T column
    customer_history['T'] = np.arange(customer_history.shape[0])
    # add cumulative transactions column
    customer_history['frequency'] = customer_history['transactions'].cumsum() \
        - 1  # first purchase is ignored
    # add t_x column
    customer_history['recency'] = customer_history.apply(
        lambda row: row['T'] if row['transactions'] != 0 else np.nan, axis=1)
    customer_history['recency'] = customer_history['recency'] \
        .fillna(method='ffill').fillna(0)
    # Calculate p_alive. (This doesn't look very Pythonian, but alright...)
    return customer_history.apply(
        lambda row: model.conditional_probability_alive(row['frequency'],
                                                        row['recency'],
                                                        row['T']), axis=1)


def _fit(minimizing_function, minimizing_function_args, iterative_fitting,
         initial_params, params_size, disp, tol=1e-8):
    """Iterative fitting function using `fmin` from scipy.optimize."""
    ll = []
    sols = []

    def _func_caller(params, func_args, function):
        return function(params, *func_args)

    if iterative_fitting <= 0:
        raise ValueError("iterative_fitting parameter should be greater "
                         "than 0 as of lifetimes v0.2.1")

    current_init_params = np.random.normal(1.0, scale=0.05, size=params_size) \
        if initial_params is None else initial_params
    total_count = 0
    while total_count < iterative_fitting:
        xopt, fopt, _, _, _ = fmin(_func_caller, current_init_params, ftol=tol,
                                   args=(minimizing_function_args,
                                         minimizing_function), disp=disp,
                                   maxiter=2000, maxfun=2000, full_output=True)
        sols.append(xopt)
        ll.append(fopt)
        total_count += 1

    argmin_ll, min_ll = min(enumerate(ll), key=lambda x: x[1])
    minimizing_params = sols[argmin_ll]

    return minimizing_params, min_ll


def _scale_time(age):
    """Create a scalar such that the maximum age is 10."""
    return 10. / age.max()


def _check_inputs(frequency, recency=None, T=None, monetary_value=None):
    if recency is not None:
        if T is not None and np.any(recency > T):
            raise ValueError("Some values in recency vector are larger than T "
                             "vector.")
        if np.any(recency[frequency == 0] != 0):
            raise ValueError("There exist non-zero recency values when "
                             "frequency is zero.")
        if np.any(recency < 0):
            raise ValueError("There exist negative recency (ex: last order "
                             "set before first order)")
    if np.sum((frequency - frequency.astype(int)) ** 2) != 0:
        raise ValueError("There exist non-integer values in the frequency "
                         "vector.")
    if monetary_value is not None and np.any(monetary_value <= 0):
        raise ValueError("There exist non-positive values in the "
                         "monetary_value vector.")
    # TODO: raise warning if np.any(freqency > T) as this means that there are
    # more order-periods than periods.


def customer_lifetime_value(transaction_prediction_model, frequency, recency,
                            T, monetary_value, discount_periods=12,
                            discount_period_length=30, discount_rate=0.01):
    """Compute the average lifetime value for a group of one or more customers.

    The default arguments presume monthly discounted lifetime values, i.e.
    discount_period_length = 30, with a one year forecast, i.e.
    discount_periods = 12.

    Parameters:
        transaction_prediction_model: The model to predict future transactions,
            literature uses Pareto/NBD but we can also use a different model,
            like BG/NBD.
        frequency: The frequency vector of customers' purchases (denoted $x$ in
            literature).
        recency: The recency vector of customers' purchases (denoted $t_x$ in
            literature).
        T: The vector of customers' age (time since first purchase).
        monetary_value: The monetary value vector of customer's purchases
            (denoted $m$ in literature).
        discount_periods: The expected lifetime for the user. Default: 12
        discount_rate: The monthly adjusted discount rate. Default: 0.01

    Returns:
        Series object with customer IDs as index and the estimated customer
        lifetime values as series' values.
    """
    df = pd.DataFrame(index=frequency.index)
    df['clv'] = 0  # initialize the clv column to zeros

    for i in range(discount_period_length,
                   (discount_periods * discount_period_length) + 1,
                   discount_period_length):
        # since the prediction of number of transactions is cumulative, we have
        # to subtract the previous periods
        expected_number_of_transactions = transaction_prediction_model \
            .predict(i, frequency, recency, T) \
            - transaction_prediction_model.predict(i - discount_period_length,
                                                   frequency, recency, T)
        # sum up the CLV estimates of all of the periods
        df['clv'] += (monetary_value * expected_number_of_transactions) \
            / (1 + discount_rate) ** (i / discount_period_length)

    return df['clv']  # return as a series
