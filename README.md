# We have use the miro and trello

1. Trello::https://trello.com/b/KWmwRcZv/designing-thinking

2. Miro::https://miro.com/app/board/uXjVN-Fzxmg=/?share_link_id=805306746210

# Velocity Prediction of the International Space Station (ISS) Using Machine Learning (Polynomial Regression) and PyEphem

*Abstract:*

This paper presents a research study on predicting the velocity of the International Space Station (ISS) using machine learning (polynomial regression) and PyEphem. The study utilizes acceleration and latitude-longitude information obtained from a sensor onboard the ISS. The choice of using these features, including the transformation of latitude-longitude into numerical values using the Haversine formula, as well as the selection of machine learning (polynomial regression) and PyEphem over alternative methods such as OpenCV, are thoroughly explained.

## 1. Introduction

For many uses in space science and exploration, a precise estimate of the International Space Station's velocity is essential. This work presents a unique method for predicting the ISS velocity using acceleration and latitude-longitude data gathered from an onboard sensor. It blends machine learning (polynomial regression) with PyEphem. In-depth discussions are held about the rationale behind the selection of PyEphem and machine learning, as well as the rejection of other approaches such as OpenCV.

## 2. Problem Statement

The objective of this study is to predict the velocity of the ISS based on the acceleration and latitude-longitude information acquired from a sensor onboard the ISS. The rationale behind selecting machine learning (polynomial regression) and PyEphem as the preferred approach is elaborated, highlighting their compatibility with the provided dataset.

## 3. Feature Selection and Transformation

### 3.1 Acceleration

Acceleration is a fundamental physical parameter that directly influences the velocity of an object. By incorporating acceleration as a feature, the machine learning model can capture the relationship between acceleration and velocity, allowing for accurate predictions.

### 3.2 Latitude-Longitude Transformation

The latitude-longitude information provides valuable spatial context. To utilize it effectively, the Haversine formula is applied to convert latitude-longitude coordinates into numerical values. This transformation enables the inclusion of geographical information as numerical features in the machine learning model.

### 3.3 Time Difference and Latitude

The machine learning model further makes advantage of latitude and time difference. Latitude is a geographical indication that may have an impact on the velocity because of the Earth's curvature, whereas time difference explains temporal fluctuations in the velocity of the International Space Station. By adding these characteristics, the model becomes more predictive.

## 4. Justification for Machine Learning (Polynomial Regression) and PyEphem

### 4.1 Machine Learning (Polynomial Regression)

Machine learning, specifically polynomial regression, is chosen because it can capture complex nonlinear relationships between the input features (acceleration, transformed latitude-longitude, time difference, and latitude) and the target label (velocity). Polynomial regression allows for the modeling of higher-order interactions, providing a flexible and powerful approach to velocity prediction.

### 4.2 PyEphem

PyEphem is employed for its capabilities in calculating celestial positions and motions. It leverages the known ISS fixed orbit and latitude-longitude information to estimate velocity accurately. PyEphem's functionality complements the analysis of ISS velocity predictions by incorporating astronomical calculations that directly affect the ISS's motion.

## 5. Comparison with Alternative Methods

The distinct problems presented by the space environment provide as justification for using PyEphem and machine learning (polynomial regression) over other approaches, including OpenCV. Since there is no trustworthy reference point in space and the Earth's surface is frequently hidden by clouds or darkness, visual-based techniques like OpenCV are less appropriate. While PyEphem provides astronomical computations that directly contribute to the study of the International Space Station (ISS) in the absence of visual clues, machine learning (polynomial regression) allows the flexibility to capture complicated correlations in the data.

## 6. Conclusion

This paper presented a comprehensive analysis of predicting the velocity of the ISS using machine learning (polynomial regression) and PyEphem. By incorporating acceleration, transformed latitude-longitude, time difference, and latitude as features, the proposed approach achieved accurate velocity predictions. The selection of machine learning (polynomial regression) and PyEphem was justified based on their ability to capture complex relationships, handle numerical transformations, and leverage astronomical calculations. Comparisons with alternative methods, such as OpenCV, demonstrated the suitability of the chosen approach for the given problem.

# Utilising Machine Learning, Computer Vision, and the Haversine Formula to Predict the International Space Station's Speed

This repository contains a project that aims to predict the speed of the International Space Station (ISS) using machine learning, computer vision, and the Haversine formula. The goal is to develop an algorithm that accurately estimates the ISS's speed based on various data sources and techniques.

## Introduction

The International Space Station (ISS) orbits around the Earth at a high speed, and accurately predicting its velocity is crucial for various scientific and operational purposes. In this project, we leverage machine learning, computer vision, and the Haversine formula to estimate the ISS's speed.

## Methodology

The project utilizes the following techniques and methodologies:

1. **Machine Learning**: Training data is collected from the ISS's real-time API, which includes altitude, latitude, longitude, and labeled speed data. We employ the random forest regression model, a supervised learning algorithm known for its excellent predictive power.

~~2. **Computer Vision**: By analyzing the changes in features between consecutive images, we can determine the ISS's speed using computer vision techniques.~~

3. **Calculating**: The Haversine formula, a great-circle distance formula, is used to calculate the ISS's speed. Our calculations assume that the ISS is located approximately 420 km from Earth.

## Results

The project's results are as follows:

- **Machine Learning**: The estimated speed results from the machine learning model were consistently within the range of 7 km/s.

~~- **Computer Vision**: The estimated speed results varied between 7 km/s and 8 km/s using computer vision techniques.~~

- **Haversine Formula**: The majority of estimated speeds were within 7 km/s, with a few falling within the range of 6 km/s to 10 km/s.

The estimated speed of the ISS is approximately 7 km/s, which closely aligns with the actual speed of the ISS.

## Conclusion

In conclusion, the algorithm developed in this project demonstrates a fair accuracy in estimating the speed of the International Space Station. However, there is still room for improvement. Future work could focus on fine-tuning the hyperparameters of the machine learning model to further enhance estimation accuracy. Additionally, incorporating sensor data and refining the Haversine formula could lead to more precise estimations. We also plan to explore the use of additional data sources to improve the accuracy of the ISS's speed estimation.
