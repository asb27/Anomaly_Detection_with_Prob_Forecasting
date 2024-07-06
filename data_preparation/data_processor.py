import os
import holidays
import numpy as np
import pandas as pd
from config_loader import ConfigLoader
from data_preparation.data_load import DataLoad

class DataProcessor:
    def __init__(self, config_loader):
        self.data_directory = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
        self.data_loader = DataLoad(self.data_directory)
        self.config = config_loader.get_config('data_processor')

    def add_time_features(self, df):
        df['day_of_week'] = df.index.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['is_weekend'] = df.index.dayofweek // 5
        df['hour_of_day'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['DayOfYear'] = df.index.dayofyear
        df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        return df

    def add_holiday_feature(self, df, year):
        data = ((date, 1) for date in holidays.country_holidays('DE', subdiv='NI', years=year))
        NI_holidays = pd.DataFrame(data, columns=['date', 'holiday'])
        NI_holidays['date'] = pd.to_datetime(NI_holidays['date'])
        holiday_dates_only = NI_holidays['date'].dt.date
        df_dates = pd.Series(df.index.date)
        is_holiday = df_dates.isin(holiday_dates_only).values
        df.loc[is_holiday, 'is_weekend'] = 1
        return df

    def interpolate_df_with_limit(self, df, method='linear', limit=24):
        df_interpolated = df.copy()
        for column in df.columns:
            is_nan = df[column].isna()
            if is_nan.sum() == 0:
                continue
            nan_groups = is_nan.ne(is_nan.shift()).cumsum()
            group_sizes = nan_groups[is_nan].value_counts()
            valid_groups = group_sizes[group_sizes > limit].index
            mask = ~nan_groups.isin(valid_groups)
            df_interpolated.loc[mask, column] = df_interpolated.loc[mask, column].interpolate(method=method)
        return df_interpolated

    def add_moving_averages(self, df):
        moving_averages = self.config.get('moving_averages', {})
        for column, windows in moving_averages.items():
            for window in windows:
                df[f'{column}_MA_{window}'] = df[column].shift(1).rolling(window=window).mean()
                #or we can use this code
                '''df['MA_shifted_no_shift'] = df['Değer'].rolling(window=4).apply(
                    lambda x: x[:-1].mean() if len(x) == 4 else pd.NA, raw=False)'''

        return df

    def add_lags(self, df):
        lags = self.config.get('lags', {})
        for column, lag_periods in lags.items():
            for lag in lag_periods:
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df

    def create_dataframe_common(self, year, datestart, dateend, sampling):
        print('dcreating dataframe')
        weather_dic = self.data_loader.readweatherall(year, datestart, dateend, sampling)
        weather_df = pd.concat(weather_dic.values(), axis=1)
        data, psum = self.data_loader.read_data('1min', year, datestart, dateend, sampling, with_hp=False)
        df = psum.to_frame(name='Consumption')
        merge_tot = pd.concat([df, weather_df], axis=1)
        merge_tot = self.add_time_features(merge_tot)
        merge_tot = self.add_holiday_feature(merge_tot, year)
        merge_tot = self.add_moving_averages(merge_tot)
        merge_tot = self.add_lags(merge_tot)
        merge_tot.drop(['hour_of_day', 'month', 'day_of_week', 'DayOfYear'], axis=1, inplace=True)

        # Config
        selected_variables = self.config['variables']
        merge_tot = merge_tot[selected_variables]
        merge_tot = merge_tot.dropna()

        return merge_tot
    def create_dataframe(self, year, datestart, dateend, sampling='15Min'):
        return self.create_dataframe_common(year, datestart, dateend, sampling)

    def create_dataframe_1min(self, year, datestart, dateend, sampling='1min'):
        df = self.create_dataframe_common(year, datestart, dateend, sampling)
        df = self.interpolate_df_with_limit(df, method='linear', limit=66)
        return df
