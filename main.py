from exif import Image
from datetime import datetime
import cv2
import math
from pathlib import Path
from math import sin, cos, sqrt, atan2, radians
import pandas as pd
import joblib
from time import sleep
from picamera import PiCamera

def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

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


def calculate_speed(image_1, image_2):
    latitude1 = get_latitude(image_1)
    longitude1 = get_longitude(image_1)
    latitude2 = get_latitude(image_2)
    longitude2 = get_longitude(image_2)
    lat_decimal1 = degree_to_decimal_lat(latitude1)
    lon_decimal1 = degree_to_decimal_lon(longitude1)
    lat_decimal2 = degree_to_decimal_lat(latitude2)
    lon_decimal2 = degree_to_decimal_lon(longitude2)
    R = 6371 + 420

    a = radians(lat_decimal2 - lat_decimal1)
    b = radians(lon_decimal2 - lon_decimal1)
    x = sin(a/2)**2 + cos(a) * cos(a) * sin(b/2)**2
    c = 2 * atan2(sqrt(x), sqrt(1-x))

    distance = R * c
    time_difference = get_time_difference(image_1, image_2)
    speed = distance / time_difference
    return speed


def predict_speed(image):
    model = joblib.load('model.pkl')
    latitude = get_latitude(image)
    longitude = get_longitude(image)
    lat_decimal = degree_to_decimal_lat(latitude)
    lon_decimal = degree_to_decimal_lon(longitude)
    predicted_speed = model.predict([[lat_decimal, lon_decimal]])  
    return predicted_speed

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)

def calculate_speed_in_kmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed


def calculate_average_speed():
    speeds = []
    for i in range(1, 42):
        image = f'image{i}.jpg'
        speed = predict_speed(image)
        speeds.append(speed)
    average_speed_prediction = sum(speeds) / len(speeds) / 1000
    print("===================================================")
    print("Average Speed (Machine Learning):", average_speed_prediction)
    
    speeds = []
    for i in range(1, 42):
        image_1 = f'image{i}.jpg'
        image_2 = f'image{i+1}.jpg'
        speed = calculate_speed(image_1, image_2)
        speeds.append(speed)
    average_speed_1_2 = sum(speeds) / len(speeds)
    print("Average Speed (Haversine Formula):", average_speed_1_2)

    image_1 = 'image12.jpg'
    image_2 = 'image13.jpg'
    time_difference = get_time_difference(image_1, image_2)
    image_1_cv, image_2_cv = convert_to_cv(image_1, image_2)
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    average_speed_feature_matching = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
    print("Average Speed (OpenCV Feature Matching):", average_speed_feature_matching)

    # Calculate weighted average speed
    total_average_speed = (0.1 * average_speed_1_2) + (0.75* average_speed_prediction) + (0.15 * average_speed_feature_matching)

    return total_average_speed
if __name__ == "__main__":
    total_average_speed = calculate_average_speed()
    total = []
    total.append(total_average_speed)
    total_average_speed_str = str(total_average_speed)
    with open('result.txt', 'x') as Doc:
        Doc.write(total_average_speed_str)
        print('Total Average Speed:', + total_average_speed)
        print("===================================================")
    