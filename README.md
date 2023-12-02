# Utilising Machine Learning, Computer Vision, and the Haversine Formula to Predict the International Space Station's Speed

This repository contains a project that aims to predict the speed of the International Space Station (ISS) using machine learning, computer vision, and the Haversine formula. The goal is to develop an algorithm that accurately estimates the ISS's speed based on various data sources and techniques.

## Introduction

The International Space Station (ISS) orbits around the Earth at a high speed, and accurately predicting its velocity is crucial for various scientific and operational purposes. In this project, we leverage machine learning, computer vision, and the Haversine formula to estimate the ISS's speed.

## Methodology

The project utilizes the following techniques and methodologies:

1. **Machine Learning**: Training data is collected from the ISS's real-time API, which includes altitude, latitude, longitude, and labeled speed data. We employ the random forest regression model, a supervised learning algorithm known for its excellent predictive power.

2. **Computer Vision**: By analyzing the changes in features between consecutive images, we can determine the ISS's speed using computer vision techniques.

3. **Calculating**: The Haversine formula, a great-circle distance formula, is used to calculate the ISS's speed. Our calculations assume that the ISS is located approximately 420 km from Earth.

## Results

The project's results are as follows:

- **Machine Learning**: The estimated speed results from the machine learning model were consistently within the range of 7 km/s.

- **Computer Vision**: The estimated speed results varied between 7 km/s and 8 km/s using computer vision techniques.

- **Haversine Formula**: The majority of estimated speeds were within 7 km/s, with a few falling within the range of 6 km/s to 10 km/s.

The estimated speed of the ISS is approximately 7 km/s, which closely aligns with the actual speed of the ISS.

## Conclusion

In conclusion, the algorithm developed in this project demonstrates a fair accuracy in estimating the speed of the International Space Station. However, there is still room for improvement. Future work could focus on fine-tuning the hyperparameters of the machine learning model to further enhance estimation accuracy. Additionally, incorporating sensor data and refining the Haversine formula could lead to more precise estimations. We also plan to explore the use of additional data sources to improve the accuracy of the ISS's speed estimation.
