import pandas
import numpy
import yfinance
# import indices


def normalize_price_average(data):
    """
    the main idea of normalization is:
        acquire the highs and lows of prices, and normalize according to them;

    pandas offer regex methods for selecting columns with certain patterns (...avg)
    :param data:
    :return:
    """
    # extract high and low from column "High" and "Low"
    # close =
    high = data["High"].max()
    low = data["Low"].min()
    all_columns = extract_price_column_name(data)
    for columns in all_columns:
        new_col = []
        for datapoints in data[columns]:
            if pandas.isna(datapoints):
                new_col.append(datapoints)
                continue
            normalized_point = normalize_formula(datapoints, high, low)
            new_col.append(normalized_point)

        data[columns] = new_col


def normalize_volume(data):
    """
    realizing the normalization of volume is regarding the batch only;
    :param data:
    :return:
    """
    volume = data["Volume"]
    v_max = volume.max()
    v_min = volume.min()
    new_volume = volume.apply(normalize_formula, args=(v_max, v_min))
    data["Volume"] = new_volume


def normalize_macd(data):
    """
    the normalization involves the following steps:
    extract out MACD's line and signal columns, standardize by the max and min
    from the two columns;
    after that will re-calculate the histogram;
    realizing the sign of MACD's valeus and histogram would matter; thus should call "find_zero_point"
    and shift the whole data by that value, to maintain the zero_points;

    :param data:
    :return:
    """
    line = data["MACD_line"]
    signal = data["MACD_signal"]
    macd_max, macd_min = _find_macd_max_min(line, signal)

    new_line = line.apply(normalize_formula, args=(macd_max, macd_min))
    new_signal = signal.apply(normalize_formula, args=(macd_max, macd_min))
    # realize: when calling pandas.Series.apply(), "datapoint", by default, is the FIRST argument to be passed;
    # here macd_max, macd_min are respectively 2nd and 3rd argument of "normalize_formula" function;

    zero_shifting = find_zero_point(macd_max, macd_min)
    new_line -= zero_shifting
    new_signal -= zero_shifting
    new_hist = new_line - new_signal

    data["MACD_line"] = new_line
    data["MACD_signal"] = new_signal
    data["MACD_hist"] = new_hist


def _find_macd_max_min(line_data, signal_data):
    macd_max = max(line_data.max(), signal_data.max())
    macd_min = min(line_data.min(), signal_data.min())
    return macd_max, macd_min


def extract_price_column_name(data):
    """
    currently price columns includes prices, MA, and EMA
    :param data:
    :return:
    """
    price_col = []
    basic_col = ["Open", "High", "Low", "Close"]
    all_col = data.columns
    for basic_columns in basic_col:
        if basic_columns in all_col:
            price_col.append(basic_columns)
    avg_col = data.filter(regex="[0-9]*avg").columns
    ema_col = data.filter(regex="EMA\_[0-9]*").columns
    price_col.extend(avg_col)
    price_col.extend(ema_col)
    return price_col


def normalize_formula(data_point, high, low):
    """
    calculation method:

    (data-low)/(high-low);

    :param high:
    :param low:
    :param data_point:
    :return: single standardized value
    """
    return (data_point - low) / (high - low)


def find_zero_point(high, low):
    """
    for finding the zero lines, as a way of preservation.
    calculates and returns the shifting deviation factor

    mechanism: determine "zero"'s normalized datapoint along with (high, low)'s deviation,
    where the value should be added to each data point to achieve shifting while preserving zero points.
    :return:
    """
    return (0 - low) / (high - low)


def normalize_WR(data, sign_reverse=True):
    """
    simply divide all data by 100; if possible will also apply sign reverse, by moving all data by 1 up.
    :param data:
    :return:
    """
    wr_columns = data.filter(regex="WR_.*").columns
    for column in wr_columns:
        new_col = (data[column] / 100) + 1
        data[column] = new_col


if __name__ == "__main__":
    read_data = pandas.read_csv("../data/spy1001-2201.csv")
    normalize_price_average(read_data)
    normalize_volume(read_data)
    normalize_macd(read_data)
    normalize_WR(read_data)
    # a = read_data["Volume"].max()
    # b = 2