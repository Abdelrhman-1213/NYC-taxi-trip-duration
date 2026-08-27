# NYC Taxi Trip Duration Prediction

An end-to-end machine learning pipeline predicting the total ride duration of taxi trips in New York City. 

This project demonstrates a complete machine learning lifecycle, from EDA and feature engineering to building production-ready Scikit-Learn pipeline.

## Project Results & Evolution
The model was built iteratively, focusing heavily on feature engineering to capture the real-world traffic patterns.

* **Baseline Model:** R2 Score: `0.066`
* **Intermediate Model:** R2 Score: `~0.620` (Added basic temporal features)
* **Final Model:** * **Test R2 Score:** `0.82`


**How I achieved this 75% performance leap:** By engineering domain-specific features rather than just throwing a more complex algorithm at the problem.

## Key Feature Engineering 

### 1. Geospatial Engineering
* **Distance:** Calculate the distance between pickup and dropoff coordinates using **Haversine** formula.

* **Bearing:** Calculated the direction (compass direction in degrees, 0-360) of the trip. Because Manhattan's grid is tilted ~29 degrees off true North, this allowed the model to differentiate between heavy avenue traffic (North/South) and cross-street traffic (East/West).
* **Geospatial Clustering:** Applied **K-Means Clustering** to the pickup/dropoff coordinates to group rides by neighborhoods. *(Note: K-Means was fitted  on the training set and only transformed on validation/test sets to prevent data leakage).*

### 2. Temporal & Domain Logic
* Extracted `dayofweek`, `dayofyear`, and `hour` to map real-world routines.
* Modeled the **NYC Rush Hour** by capturing the distinct traffic spikes that occur specifically between 2:00 PM (14:00) and 5:00 PM (17:00) on weekdays versus weekends.
* Engineered a `weekend` as a flag to identify rides occurring on weekends.
* Engineered a `rush_hour` as a flag to identify rides occurring on rush hours.

* Extracted `cyclical features` from the `hour` and `dayofweek`
## Machine Learning Pipeline
### 1. Data preprocessing:
* applied logarithmic transformation (`np.log1p`) of **distance** and **target_variable** due to heavy right-skewness

* **Numeric Features:** Scaled using `StandardScaler` to preserve spatial differences in GPS coordinates without squishing outliers (unlike MinMax).
* **Categorical Features:** Encoded using `OneHotEncoder(handle_unknown='ignore')`.
* didn't apply scaling on `cyclical_features` , `weekend`,`rush_hour`

### 2. Used a XGBRegressor:
* Chosen for its ability to handle complex, overlapping patterns within the dataset that simpler linear models(like Ridge) cannot capture.
## Project Structure
```text
nyc-taxi-trip-duration/
│
├── data/                   # Ignored in Git (Raw csv files)
├── notebooks/               # Jupyter notebooks 
|   ├── EDA_Training_Data.ipynb # EDA and visualizations
│   ├── Base_line_model.ipynb   #Base line model
├── src/                    # Modular Python scripts for production
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── requirements.txt        # Python dependencies
├── .gitignore              # Environment and data protection
└── README.md               # Project documentation