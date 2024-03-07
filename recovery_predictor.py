import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class recoveryPredictor():
    def __init__(self, activity, sleep, heart_rate):
        self.activity = activity
        self.sleep = sleep
        self.detailed_sleep_data = None
        self.heart_rate = heart_rate
        self.recovery_data = None
        self.heart_rate_sleep_data = None


        # Pre processing
        self.sleep_preprocess()
        self.heart_rate_preprocess()
        self.get_recovery_data()

        # Normalizeing
        self.norm_recovery_data()


        # Enginerring features
        self.recovery_score()
        
    def fit(self):
      
        pass

    def sleep_preprocess(self):
        # Drop all emptyp cells
        self.sleep.dropna(inplace=True)

        # Group by 'Id' and 'logId' directly from the original sleep DataFrame
        self.detailed_sleep_data = self.sleep.groupby(['Id', 'logId']).agg(
            start_time=('date', 'min'),
            end_time=('date', 'max'),
            total_minutes=('value', 'count'),
            asleep_minutes=('value', lambda x: (x == 1).sum()),
            restless_minutes=('value', lambda x: (x == 2).sum()),
            awake_minutes=('value', lambda x: (x == 3).sum())
        ).reset_index()

    def heart_rate_preprocess(self):
        # Drop all emptyp cells
        self.heart_rate.dropna(inplace=True)

        # Convert Time to pd time format
        self.heart_rate['Time'] = pd.to_datetime(self.heart_rate['Time'])

    def get_recovery_data(self):
        # Apply the filter_heartrate_from_sleep function to each row of detailed_sleep_data
        self.heart_rate_sleep_data = self.detailed_sleep_data.apply(lambda row: self.filter_heartrate_from_sleep(row, self.heart_rate), axis=1)

        # Concatenate the filtered heart rate data into a single DataFrame
        self.heart_rate_sleep_data = pd.concat(self.heart_rate_sleep_data.tolist())

        # Convert Time to pd time format
        self.heart_rate_sleep_data['Time']= self.heart_rate_sleep_data['Time'].dt.date

        # Lets find the lowers observed heartrate during every night for every user and call this resting heart rate
        self.find_resting_heart_rate()

        self.build_recovery_data()

    def filter_heartrate_from_sleep(self, sleep_data_row, heart_rate_data):
        filter_data = heart_rate_data[(heart_rate_data['Time'] >= sleep_data_row['start_time']) & 
                                        (heart_rate_data['Time'] <= sleep_data_row['end_time'])]
        return filter_data

    def find_resting_heart_rate(self):
        self.heart_rate_sleep_data = self.heart_rate_sleep_data.groupby(['Id','Time']).agg(
            resting_heart_rate = ('Value', 'min')
        ).reset_index()

        self.heart_rate_sleep_data.rename(columns={'Time': 'date'}, inplace=True)

    def build_recovery_data(self):
        self.detailed_sleep_data['start_time'] = pd.to_datetime(self.detailed_sleep_data['start_time'])
        self.detailed_sleep_data['start_time'] = pd.to_datetime(self.detailed_sleep_data['start_time'])
    
        temp_sleep = self.detailed_sleep_data
        temp_sleep.rename(columns={'start_time': 'date'}, inplace=True)
        temp_sleep.drop(columns=['end_time'], inplace=True)
        temp_sleep.drop(columns=['total_minutes'], inplace=True)
        temp_sleep.drop(columns=['restless_minutes'], inplace=True)
        temp_sleep.drop(columns=['awake_minutes'], inplace=True)
        temp_sleep['date'] = pd.to_datetime(temp_sleep['date']).dt.date
    
        self.recovery_data = pd.merge(temp_sleep, self.heart_rate_sleep_data, on=['Id', 'date'], how='inner') 

    def norm_recovery_data(self):
        scaler = MinMaxScaler()
        self.recovery_data[['asleep_minutes', 'resting_heart_rate']] = scaler.fit_transform(self.recovery_data[['asleep_minutes', 'resting_heart_rate']])

    def calculate_recovery_score(self, asleep_min_norm, rhr_norm):
        rhr_weight = 0.6
        asleep_weight = 0.4

        recovery_score = (rhr_weight * rhr_norm) + (asleep_weight * (1 - asleep_min_norm))

        recovery_score = recovery_score * 100

        return recovery_score

    def recovery_score(self):
        self.recovery_data['Recovery_score'] = self.calculate_recovery_score(self.recovery_data['asleep_minutes'], self.recovery_data['resting_heart_rate'])

def main():
    activity_df = pd.read_csv("data/dailyActivity_merged.csv")
    calories_df = pd.read_csv("data/hourlyCalories_merged.csv")
    sleep_df = pd.read_csv("data/minuteSleep_merged.csv")
    heart_rate_df = pd.read_csv("data/heartrate_seconds_merged.csv")



    model = recoveryPredictor(activity_df, sleep_df, heart_rate_df)

    print(model.recovery_data)



if __name__ == "__main__":
    main()