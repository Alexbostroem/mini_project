import pandas as pd


class recoveryPredictor():
    def __init__(self, activity, sleep, heart_rate):
        self.activity = activity
        self.sleep = sleep
        self.detailed_sleep_data = None
        self.heart_rate = heart_rate
        self.recovery_data = None

        # Pre processing
        self.recovery_preprocess()
        self.heart_rate_preprocess()

        # Enginerring features

        #N

    def fit(self):
      
        pass

    def recovery_preprocess(self):
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
        self.heart_rate['Time'] = pd.to_datetime(self.heart_rate['Time'])






def main():
    activity_df = pd.read_csv("data/dailyActivity_merged.csv")
    calories_df = pd.read_csv("data/hourlyCalories_merged.csv")
    sleep_df = pd.read_csv("data/minuteSleep_merged.csv")
    heart_rate_df = pd.read_csv("data/heartrate_seconds_merged.csv")



    model = recoveryPredictor(activity_df,sleep_df, heart_rate_df)




if __name__ == "__main__":
    main()