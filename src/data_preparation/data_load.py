import os
import pandas as pd

class DataLoad:
    """
    The DataLoad class is designed to load data from the hdf5 files. It is used by the DataProcessor class to load,
    filter and resample data.

    The data reloads for each model. In fact, once the data is read and cached,
    other models can use this cache to improve performance.
    """
    def __init__(self, data_directory):
        self.data_directory = data_directory
        '''self.date_ranges = {
            2019: (pd.to_datetime('2019-01-01 00:00' ), pd.to_datetime('2019-12-31 23:59')),
            2020: (pd.to_datetime('2020-01-01 00:00:00'), pd.to_datetime('2020-12-31 23:59:59'))
            
        } '''

    def read_hdf(self, file_path, key):
        try:
            return pd.read_hdf(os.path.join(self.data_directory, file_path), key)
        except Exception as e:
            print(f"Error reading {key} from {file_path}: {e}")
            raise

    def readdata(self, timeres='1min', year=2019, datestart=None, dateend=None, sampling='5min', with_hp=False):
        file_path = f"{year}_data_{timeres}.hdf5"
        exclude = self.get_exclusions(year)
        data = {}
        start_index, end_index = self.get_indices(datestart, dateend, timeres)

        for i in range(3, 41):
            if i not in exclude:
                household_data = self.read_hdf(file_path, f"NO_PV/SFH{i}/HOUSEHOLD/table")
                if not household_data.empty:
                    household_data = self.filter_data_by_index(household_data, start_index, end_index)
                    hh_power = household_data["P_TOT"]
                    total_power = self.add_heat_pump_power(household_data, file_path, i, start_index, end_index) if with_hp else hh_power
                    data[f"HH{i}"] = total_power.resample(sampling).mean()

        return pd.DataFrame(data), pd.DataFrame(data).sum(axis=1)

    def get_exclusions(self, year):
        base_exclusion = {13, 15, 26, 33}
        additional_exclusion = {
            2019: {6, 8, 10, 11, 17, 23, 24, 25, 31, 34, 37, 40},
            2020: {6, 8, 10, 11, 17, 23, 24, 25, 31, 34, 37, 40},
        }
        return base_exclusion | additional_exclusion.get(year, set())

    def get_indices(self, datestart, dateend, timeres):
        start_index = pd.to_datetime(datestart).floor(timeres) if datestart else None
        end_index = pd.to_datetime(dateend).floor(timeres) if dateend else None
        return start_index, end_index

    def filter_data_by_index(self, data, start_index, end_index):
        if start_index:
            data = data[start_index:]
        if end_index:
            data = data[:end_index]
        return data

    def add_heat_pump_power(self, household_data, file_path, i, start_index, end_index):
        heat_pump_data = self.read_hdf(file_path, f"NO_PV/SFH{i}/HEATPUMP/table")
        return household_data["P_TOT"] + heat_pump_data["P_TOT"] if not heat_pump_data.empty else household_data["P_TOT"]



    def readweatherall(self, year, datestart=None, dateend=None, sampling='5min'):
        file = f"{self.data_directory}/{year}_weather.hdf5"
        weather = []
        keys = ['temperature', 'solar_irradiance', 'wind', 'humidity', 'pressure', 'apparent_temperature', 'precipitation',
                'pr_precipitation', 'wind_direction', 'wind_gust']
        names = [

            'WEATHER_TEMPERATURE_TOTAL',
            'WEATHER_SOLAR_IRRADIANCE_GLOBAL',
            'WEATHER_WIND_SPEED_TOTAL',
            'WEATHER_RELATIVE_HUMIDITY_TOTAL', 'WEATHER_ATMOSPHERIC_PRESSURE_TOTAL',
            'WEATHER_APPARENT_TEMPERATURE_TOTAL',
            'WEATHER_PRECIPITATION_RATE_TOTAL',
            'WEATHER_PROBABILITY_OF_PRECIPITATION_TOTAL',
            'WEATHER_WIND_DIRECTION_TOTAL',
            'WEATHER_WIND_GUST_SPEED_TOTAL']

        for key, name in zip(keys, names):
            raw = pd.read_hdf(file, f"WEATHER_SERVICE/IN/{name}")  # temperature
            raw = raw.tz_localize('UTC').tz_convert('Europe/Berlin')  # convert to UTC+1
            raw.index = raw.index.tz_localize(None)

            cleaned = raw.groupby(level=0).mean()
            wea = cleaned.resample(sampling).mean()

            if datestart and dateend:
                wea = wea.loc[datestart:dateend]

            # Reindex with the complete range of timestamps
            complete_range = pd.date_range(start=datestart, end=dateend, freq=sampling)

            wea = wea.reindex(complete_range)

            timestamps = [pd.Timestamp(t[0]) for t in wea.items()]
            data = pd.DataFrame([t[1] for t in wea.items()])
            all = pd.DataFrame({'Timestamp': timestamps, key: data.values.flatten()}).set_index('Timestamp')
            all.index = pd.to_datetime(all.index)
            all = all.resample(sampling).mean()
            weather.append(all)
        weather_dict = {
            'temperature': weather[0],
            'solar_irradiance': weather[1],
            'wind': weather[2],
            'humidity': weather[3],
            'pressure': weather[4],
            'apparent_temperature': weather[5],
            'precipitation': weather[6],
            'pr_precipitation': weather[7],
            'wind_direction': weather[8],
            'wind_gust': weather[9]
        }

        return weather_dict


    def read_data(self, timeres='1min', year=None, datestart=None, dateend=None, sampling='5min', with_hp=False):  # dateend seconds
        file = f"{self.data_directory}/{year}_data_{timeres}.hdf5"

        samplerate = {
            "15min": 60 * 15,
            "1min": 60,
            "10s": 10,
        }[timeres]
        exclude = {13, 15, 26, 33}  # PV
        exclude |= {
            2019: {6, 8, 10, 11,  17, 23, 24, 25, 31, 34, 37, 40},
            # availability<99% ; [34] unusual behavior. metadata analysis:[6, 17, 25, 31, 37, 40]. 24 no availability. How unusual behaviour for 34 determined?
            2020: {6, 8, 10, 11,  17, 23, 24, 25, 31, 34, 37, 40},
            # availability<99%. 24, 25 no availability.metadata analysis: [6, 8, 10, 11, 17, 23, 31, 35]
        }[year]
        data = {}
        if datestart:
            start_index = pd.to_datetime(datestart).floor(timeres)
        else:
            start_index = 0
        if dateend:
            end_index = pd.to_datetime(dateend).floor(timeres)
        else:
            end_index = None

        for i in range(3, 41):
            if i in exclude:
                continue
            d = pd.read_hdf(file, f"NO_PV/SFH{i}/HOUSEHOLD/table")
            d.index = pd.to_datetime(d["index"], unit='s')

            if start_index:
                d = d[start_index:]
            if end_index:
                d = d[:end_index]
            hh = d["P_TOT"]
            if with_hp:
                hp = pd.read_hdf(file, f"NO_PV/SFH{i}/HEATPUMP/table")["P_TOT"]

                hp.index = d.index
                if start_index:
                    hp = hp[start_index:]
                if end_index:
                    hp = hp[:end_index]
                p = hh + hp
            else:
                p = hh

            # Resample to x-minute intervals
            p = p.resample(sampling).mean()

            data[f"HH{i}"] = p

        df = pd.DataFrame(data)

        psum = df.sum(axis=1)
        return df, psum
