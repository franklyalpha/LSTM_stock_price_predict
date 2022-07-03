# this file only contains code for common operations;
# when an operation is required, should copy-paste the code into python console;
# can also define functions here, and then import this file into console to use those methods implemented
import typing
import pandas
import csv
import yfinance
import numpy


def drop_col(data, column_name_str, axis=1):
    """

    :param data:
    :param column_name_str:
    :param axis: default is 1, which is column
    :return:
    """
    return data.drop(column_name_str, axis)


def remove_rows_w_missing(data, save_dest=None):
    """
    remove all rows with missing values, to ensure all remaining rows are complete. (This is to
    eliminate the effect of first few data used for calculating averages or other indices, which have missing
    values not useful for training model)
    :param data:
    :return:
    """
    data.dropna(inplace=True)
    if save_dest is not None:
        data.to_csv("../data/spy1001-2201.csv")


# https://realpython.com/python-csv/

class Indices:
    """
    the purpose of using indices is to fill in the table with another column, with values calculated as
    the property of the current indices;
    """

    def __init__(self, input_panda_data, column_name=""):
        self.panda_data = input_panda_data
        self.row_count = len(input_panda_data)
        self.new_col = []
        self.col_name = column_name

    def calculate_column(self):
        raise NotImplemented

    def update_data(self, new_data):
        assert len(new_data) > 0
        self.panda_data = new_data
        self.row_count = len(new_data)

    def append_col(self, save_dest=None, data=None):
        if data is not None:
            data[self.col_name] = self.new_col
            self._save(save_dest, data)
        else:
            self.panda_data[self.col_name] = self.new_col
            self._save(save_dest, self.panda_data)

    def _save(self, save_dest, data):
        if save_dest is not None:
            data.to_csv(save_dest)

    def return_col_name(self):
        return self.col_name

    def return_data(self):
        """
        the returned data is mutable;
        :return:
        """
        return self.panda_data


class EMA(Indices):
    """
    Exponential moving average, or expma, or ema
    """

    def __init__(self, hyperparam_lst, panda_data):
        """

        :param hyperparam_lst: index 0 is the period;
        :param panda_data:
        """
        self.period = hyperparam_lst[0]
        super().__init__(panda_data, "EMA_" + str(self.period))

    def calculate_column(self):
        close_price = self.panda_data["Close"]
        self.new_col = list(close_price.ewm(
            span=self.period, adjust=False, min_periods=self.period).mean())


class WR(Indices):
    def __init__(self, hyperparam_lst, data):
        """

        :param hyperparam_lst: index [0] is the period
        :param data:
        """
        self.period = hyperparam_lst[0]
        super().__init__(data, "WR_" + str(self.period))

        self.close_price = self.panda_data['Close']

    def calculate_column(self):
        """
        :return:
        """
        self.new_col.extend([numpy.nan] * self.period)

        for recorded_row in range(self.period, self.row_count):
            avg = self._calculate_single_row(recorded_row - self.period,
                                             recorded_row)
            self.new_col.append(avg)

    def _calculate_single_row(self, low, high):
        high_ = self.panda_data["High"].iloc[low:high].max()
        low_ = self.panda_data["Low"].iloc[low:high].min()

        curr_close = self.close_price[high - 1]
        return -100 * (high_ - curr_close) / (high_ - low_)


class MACD:
    """
    this class acts as a facade, for integrating all components forming this indices.
    """

    def __init__(self, data, line_param, sig_param):
        self.panda_data = data
        self.line = None
        self.signal = None
        self.diff = None
        self.line_param = line_param
        self.sig_param = sig_param

    def sequential_operation(self):
        """
        first calculate MACD line, and then considering this new data, calculate its signal line;
        finally find histogram_difference based on previous two lines calculated;
        :return:
        """
        self.line = MACD_line(self.line_param, self.panda_data)
        self.line.calculate_column()
        macd_line = self.line.new_col.copy()
        self.signal = MACD_signal(self.sig_param, macd_line)
        self.signal.calculate_column()
        signal_line = self.signal.new_col.copy()
        self.diff = MACD_histogram([], (macd_line, signal_line))
        self.diff.calculate_column()

    def append_all_col(self, save_dest=None, data=None):
        assert self.line is not None
        assert self.signal is not None
        assert self.diff is not None
        data_to_append = self.panda_data if data is None else data
        self.line.append_col(None, data_to_append)
        self.signal.append_col(None, data_to_append)
        self.diff.append_col(save_dest, data_to_append)


class MACD_line(Indices):
    def __init__(self, hyperparam_lst, data):
        """
        :param hyperparam_lst: index 0 is shorter-term line, index 1 is long-term line
                        index 0 is usually 12, and index 1 is usually 26
        :param data:
        """
        super().__init__(data, "MACD_line")
        self.lower_p = hyperparam_lst[0]
        self.higher_p = hyperparam_lst[1]

    def calculate_column(self):
        """
        calculates MACD line itself
        :return:
        """
        lower_EMA = self.panda_data["Close"].ewm(
            span=self.lower_p, adjust=False, min_periods=self.lower_p).mean()
        higher_EMA = self.panda_data["Close"].ewm(
            span=self.higher_p, adjust=False, min_periods=self.higher_p).mean()
        self.new_col.extend(list(lower_EMA - higher_EMA))


class MACD_signal(Indices):
    def __init__(self, hyperparam_lst, data):
        """

        :param hyperparam_lst: index 0 is usually 9;
        :param data:
        """
        super().__init__(data, "MACD_signal")
        self.signal_p = hyperparam_lst[0]
        self.panda_data = pandas.Series(self.panda_data)

    def calculate_column(self):
        """
        calculates the signal line of MACD
        :return:
        """

        self.new_col = list(self.panda_data.ewm(
            span=self.signal_p, adjust=False, min_periods=self.signal_p).mean())


class MACD_histogram(Indices):
    def __init__(self, hyperparam_lst, data):
        """
        :param hyperparam_lst: the list currently is not required;
        :param data: a tuple, where index 0 is macd line, and index 1 is signal line
        """
        super().__init__(data, "MACD_hist")
        self.line = pandas.Series(data[0])
        self.signal = pandas.Series(data[1])

    def calculate_column(self):
        self.new_col = list(self.line - self.signal)


class DayAvg(Indices):
    def __init__(self, hyperparam_lst, data):
        """

        :param hyperparam_lst:
        index 0 is: the period; 30 indicates 30-period
        index 1 is:

        """
        column_name = str(hyperparam_lst[0]) + "avg"
        # if this line is modified, should also modify code for normalizing
        #  in file "normalization"

        super().__init__(data, column_name)
        self.period = hyperparam_lst[0]
        self.closed_price = self.panda_data['Close']

    def calculate_column(self):
        # the first self.period many columns should be empty, and finally
        # only those data with complete average lines would be extracted.
        self.new_col.extend([numpy.nan] * self.period)

        for recorded_row in range(self.period, self.row_count):
            avg = self._calculate_single_row(recorded_row - self.period,
                                             recorded_row)
            self.new_col.append(avg)

    def _calculate_single_row(self, low, high):
        return self.closed_price.iloc[low:high].mean()


def add_all_indices_and_save(d_data, save_dest):
    avg10 = DayAvg([10], d_data)
    avg10.calculate_column()
    # avg60.append_col(data=d_data)
    avg10.append_col(data=d_data)

    avg20 = DayAvg([20], d_data)
    avg20.calculate_column()
    # avg60.append_col(data=d_data)
    avg20.append_col(data=d_data)

    avg35 = DayAvg([10], d_data)
    avg35.calculate_column()
    # avg60.append_col(data=d_data)
    avg35.append_col(data=d_data)

    avg60 = DayAvg([60], d_data)
    avg60.calculate_column()
    # avg60.append_col(data=d_data)
    avg60.append_col(data=d_data)

    new_macd = MACD(d_data, [12, 26], [9])
    new_macd.sequential_operation()
    new_macd.append_all_col(data=d_data)

    wr14 = WR([14], d_data)
    wr14.calculate_column()
    # wr14.append_col("data/spy1001-2201.csv")
    wr14.append_col(data=d_data)

    EMA12 = EMA([12], d_data)
    EMA12.calculate_column()
    EMA12.append_col(data=d_data)
    # EMA12.append_col("data/spy1001-2201.csv")

    EMA50 = EMA([50], d_data)
    EMA50.calculate_column()
    EMA50.append_col(data=d_data)
    # EMA50.append_col("../data/spy1001-2201.csv")
    remove_rows_w_missing(d_data, save_dest)


if __name__ == "__main__":
    d_data = yfinance.download("^IXIC", start="1990-01-01", end="2022-01-01")
    add_all_indices_and_save(d_data, "../data/spy1001-2201.csv")

