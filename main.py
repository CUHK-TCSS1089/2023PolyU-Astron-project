# importing necessary libraries
from exif import Image
from datetime import datetime
from pathlib import Path
from math import sin, cos, sqrt, radians, asin
import pandas as pd
from picamera import PiCamera
from sense_hat import SenseHat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
# loading training data from csv to dataframe
data = pd.read_csv('speeds1.csv')
df = pd.DataFrame(data)


accels = []
time = []
long =[]
lati = []
pyths=[]
distances =[]
aas =[]
# Calculating acceleration, time, distance, etc. for training model
for i in range(len(df)-1):
    # Calculating acceleration
    accel = (df['speed'][i] - df['speed'][i+1])/(df['diff_time'][i])
    
    # Calculating latitude and longitude differences
    lat = df['lat'][i] - df['lat_1'][i]
    lon = df['long'][i] - df['long_1'][i]
    
    # Converting latitude and longitude differences to radians for further calculations
    dlat = radians(lat)
    dlon = radians(lon)
    
    # Calculating the Pythagorean distance between latitude and longitude differences
    pyth = np.sqrt(lat**2 + lon**2)
    
    # Calculating distance using haversine formula
    a = sin(dlat/2)**2 + cos(radians(df['lat'][i])) * cos(radians(df['lat_1'][i])) *sin(dlon/2)**2#In our project, becuase unknow the exactly distance between earth to ISS , we choose this put in ML because it do not use the distance between earth to ISS 
    distance=2*asin(sqrt(a))*(6371+420)*1000    
    # Appending the calculated values to respective lists
    accels.append(accel)
    time.append(df['diff_time'][i])
    long.append(lon)
    lati.append(lat)
    pyths.append(pyth)
    distances.append(distance)
    aas.append(a)
    distances.append(distance)
    
# Creating dataframes for calculated values
adf = pd.DataFrame(accels)
adf.rename(columns={0:'accels'},inplace=True)
tdf = pd.DataFrame(time)
tdf.rename(columns={0:'time'},inplace=True)
latit = pd.DataFrame(lati)
latit.rename(columns={0:'latit'},inplace=True)
longi = pd.DataFrame(long)
longi.rename(columns={0:'longi'},inplace=True)
pyt = pd.DataFrame(pyths)
pyt.rename(columns={0:'pyth'},inplace=True)
distanc = pd.DataFrame(distances)
distanc.rename(columns={0:'distance'},inplace=True)
asss = pd.DataFrame(aas)
asss.rename(columns={0:'a'},inplace=True)

# Modeling Polynomial Regression
class PolynomialRegressionModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()

    def fit(self, features, labels):
        features_poly = self.poly.fit_transform(features)
        self.model.fit(features_poly, labels)

    def predict(self, features):
        features_poly = self.poly.transform(features)
        return self.model.predict(features_poly)

    def evaluate(self, features, labels):
        features_poly = self.poly.transform(features)
        predictions = self.model.predict(features_poly)
        mse = mean_squared_error(labels, predictions)
        return mse
    
# creating features and labels for the model
fe = adf.join(pd.DataFrame(df['lat']))
fe = fe.join(asss)
#fe = adf.join(tdf)
features = fe
labels = df['speed'][:len(df)-1]
# fitting the model
model = PolynomialRegressionModel(degree=5)
model.fit(features, labels)

# functions for handling image data
def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time


def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


def get_latitude(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        latitude = img.get('gps_latitude')
    return latitude


def get_longitude(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        longitude = img.get('gps_longitude')
    return longitude


def degree_to_decimal_lat(latitude):
    latitude_decimal = float(list(latitude)[0]) + float(list(latitude)[1])/60 + float(list(latitude)[2])/(60*60)
    return latitude_decimal


def degree_to_decimal_lon(longitude):
    longitude_decimal = float(list(longitude)[0]) + float(list(longitude)[1])/60 + float(list(longitude)[2])/(60*60)
    return longitude_decimal


def calculate_a(image_1, image_2):#Haversine formula
    latitude1 = get_latitude(image_1)
    longitude1 = get_longitude(image_1)
    latitude2 = get_latitude(image_2)
    longitude2 = get_longitude(image_2)
    lat_decimal1 = degree_to_decimal_lat(latitude1)
    lon_decimal1 = degree_to_decimal_lon(longitude1)
    lat_decimal2 = degree_to_decimal_lat(latitude2)
    lon_decimal2 = degree_to_decimal_lon(longitude2)
    dlat = radians(lat_decimal2 - lat_decimal1)
    dlon = radians(lon_decimal2 - lon_decimal1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat_decimal1)) * cos(radians(lat_decimal1)) * sin(dlon / 2) ** 2#In our project, becuase unknow the exactly distance between earth to ISS , we choose this no the full formula because it do not use the distance between earth to ISS 
    time_difference = get_time_difference(image_1, image_2)
    return a

def calculate_accel():
    sense = SenseHat()
    sense.clear()
    acceleration = sense.get_accelerometer_raw()
    x = acceleration['x']
    y = acceleration['y']
    z = acceleration['z']
    accel = sqrt(x**2 + y**2 + z**2)# Fing vector acceleration
    return accel



def predict_speed(image_1,image_2):
    latitude = get_latitude(image_1)
    lat_decimal = degree_to_decimal_lat(latitude)
    predicted_speed = model.predict([[calculate_accel(), lat_decimal, calculate_a(image_1, image_2)]])#ML prediction
    return predicted_speed

# main function to calculate average speed
def calculate_average_speed():
    speeds = []
    for i in range(1, 42):
        image_1 = f'image{i}.jpg'
        image_2 = f'image{i+1}.jpg'
        speed = predict_speed(image_1,image_2)
        speeds.append(speed)
    average_speed_prediction = sum(speeds) / len(speeds) / 1000
    print("===================================================")
    print("Average Speed (Machine Learning):", average_speed_prediction)


    return average_speed_prediction
# main execution
if __name__ == "__main__":
    for i in range(1, 42):
        camera = PiCamera()
        camera.resolution = (1, 1)
        base_folder = Path(__file__).parent.resolve()
        print('got')
    total_average_speed = calculate_average_speed()
    total = []
    total.append(total_average_speed)
    total_average_speed_str = str(total_average_speed)
    # writing results to a file
    with open('result.txt', 'x') as Doc:
        Doc.write(total_average_speed_str)
        print('Total Average Speed:', + total_average_speed)
        print("===================================================")
    