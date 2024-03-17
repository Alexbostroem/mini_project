import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV


class recoveryPredictor():
    def __init__(self, activity, sleep, heart_rate):
        self.activity = activity
        self.sleep = sleep
        self.detailed_sleep_data = None
        self.heart_rate = heart_rate
        self.recovery_data = None
        self.heart_rate_sleep_data = None
        self.model_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.linear_regression = LinearRegression()


        # Pre processing
        self.sleep_preprocess()
        self.heart_rate_preprocess()
        self.get_recovery_data()
        self.norm_recovery_data()

        # Enginerring features
        self.recovery_score()

        self.make_model_data()

        
    def fit(self):
        X = self.model_data.drop(['Id', 'date','asleep_minutes' ,'resting_heart_rate' ,'Recovery_score'], axis=1)
        y = self.model_data['Recovery_score']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)


        param_grid = {
            'fit_intercept': [True, False],  # Whether to calculate the intercept for this modelc
        }

        grid_search = GridSearchCV(estimator=self.linear_regression, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(self.X_train, self.y_train)

        print("Best parameters found:", grid_search.best_params_)

        self.linear_regression = grid_search.best_estimator_

        self.linear_regression.fit(self.X_train, self.y_train)

        
    def predict(self):
        y_predicted = self.linear_regression.predict(self.X_test)
        return y_predicted
    

    def score(self):
        mse = mean_squared_error(self.y_test, self.predict())
        print("Mean Squared Error:", mse)
        

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

        # Lets find the lowest observed heartrate during every night for every user and call this resting heart rate
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

        recovery_score = recovery_score

        return recovery_score

    def recovery_score(self):
        self.recovery_data['Recovery_score'] = self.calculate_recovery_score(self.recovery_data['asleep_minutes'], self.recovery_data['resting_heart_rate'])

    def make_model_data(self):
       self.activity.rename(columns={'ActivityDate': 'date'}, inplace=True)
       self.activity['date'] = pd.to_datetime(self.activity['date']).dt.date


       self.model_data = pd.merge(self.activity, self.recovery_data, on=['Id', 'date'], how='inner')


        # Columns to normalize
       columns_to_normalize = ['TotalSteps', 'TotalDistance', 'TrackerDistance', 'LoggedActivitiesDistance', 
                            'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance', 
                            'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 
                            'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories']

       scaler = MinMaxScaler()

       self.model_data[columns_to_normalize] = scaler.fit_transform(self.model_data[columns_to_normalize])

       self.model_data.drop(columns=['logId'], inplace=True)

       self.model_data.to_excel("output.xlsx", index=False)
    
    def visualize_results(self,y_predicted):
        # Scatter plot of Actual vs. Predicted values
        plt.figure(figsize=(8,6))
        plt.scatter(self.y_test, y_predicted, alpha=0.5)
        plt.xlabel('Actual Recovery Score')
        plt.ylabel('Predicted Recovery Score')
        plt.title('Actual vs. Predicted Recovery Score')
        plt.show()
        
          # Residual Plot
        plt.figure(figsize=(8, 6))
        sns.residplot(x=y_predicted, y=self.y_test - y_predicted, lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.xlabel('Predicted Recovery Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.axhline(y=0, color='k', linestyle='--')  # Add a horizontal line at y=0
        plt.show()

        # Distribution of Residuals
        plt.figure(figsize=(8, 6))
        sns.histplot(self.y_test - y_predicted, kde=True)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.show()


def main():
    activity_df = pd.read_csv("data/dailyActivity_merged.csv")
    sleep_df = pd.read_csv("data/minuteSleep_merged.csv")
    heart_rate_df = pd.read_csv("data/heartrate_seconds_merged.csv")



    model = recoveryPredictor(activity_df, sleep_df, heart_rate_df)
    model.fit()
    y_predicted = model.predict()
    model.score()
    model.visualize_results(y_predicted)
 
 



if __name__ == "__main__":
    main()