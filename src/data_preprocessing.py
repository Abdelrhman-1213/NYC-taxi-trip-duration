import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,FunctionTransformer
from sklearn.cluster import KMeans


def scale(num_features,cat_features,log_features):
    ct=ColumnTransformer(transformers=[
        ('scalar',StandardScaler(),num_features) ,
        ('ohe',OneHotEncoder(handle_unknown='ignore'),cat_features),
        ("log",FunctionTransformer(np.log1p),log_features)
    ],remainder='passthrough')
    return ct


def positional_(df:pd.DataFrame,kmean_pickup=None,kmean_dropoff=None):
    if kmean_pickup is None or kmean_dropoff is None:
        kmean_pickup=KMeans(n_clusters=5,n_init=10)
        kmean_pickup.fit(X=df[['pickup_longitude','pickup_latitude']])
        df['pickup_position']=kmean_pickup.predict(X=df[['pickup_longitude','pickup_latitude']])


        kmean_dropoff=KMeans(n_clusters=5,n_init=10)
        kmean_dropoff.fit(X=df[['dropoff_longitude','dropoff_latitude']])
        df['dropoff_position']=kmean_dropoff.predict(X=df[['dropoff_longitude','dropoff_latitude']])
        
        return kmean_pickup,kmean_dropoff
    else:
        pickup_position=kmean_pickup.predict(df[['pickup_longitude','pickup_latitude']])
        dropoff_position=kmean_dropoff.predict(X=df[['dropoff_longitude','dropoff_latitude']])
        return pickup_position,dropoff_position

    


def get_direction(df:pd.DataFrame):
    #to get the bearing use the formula:
    #θ = atan2(sin(Δλ) * cos(φ₂), cos(φ₁) * sin(φ₂) - sin(φ₁) * cos(φ₂) * cos(Δλ))
    # Where:
    # φ₁, φ₂ = latitude 1 and latitude 2 (in radians)
    # λ₁, λ₂ = longitude 1 and longitude 2 (in radians)
    # Δλ = difference between longitudes (λ₂ - λ₁)
    # θ = initial bearing direction (in radians)
    lat_1=np.radians(df['pickup_latitude'])
    lon_1=np.radians(df['pickup_longitude'])
    lat_2=np.radians(df['dropoff_latitude'])
    lon_2=np.radians(df['dropoff_longitude'])

    alpha_lon=lon_2-lon_1
    x=np.cos(lat_2)*np.sin(alpha_lon)
    y=np.cos(lat_1)*np.sin(lat_2)-np.sin(lat_1)*np.cos(lat_2)*np.cos(alpha_lon)
    df['direction']=np.atan2(x,y)
    

def get_distance(df:pd.DataFrame):
    #we calculate the distance using the Haversine formula:
        #a = sin²(Δφ / 2) + cos(φ₁) * cos(φ₂) * sin²(Δλ / 2)
        # c = 2 * atan2(√a, √(1−a))
        # d = R * c
        # Where:
        # φ₁, φ₂ = latitude 1 and latitude 2 (in radians)
        # λ₁, λ₂ = longitude 1 and longitude 2 (in radians)
        # Δφ = difference between latitudes (φ₂ - φ₁)
        # Δλ = difference between longitudes (λ₂ - λ₁)
        # R = Earth's radius (6,371 km or 3,959 miles)
        # d = distance between the two points
    R=6371.0
    alpha_latitude=np.radians(df['dropoff_latitude']) - np.radians(df['pickup_latitude'])
    alpha_longitude =np.radians(df['dropoff_longitude'])-np.radians(df['pickup_longitude'])

    a=(np.sin(alpha_latitude/2.0))**2 + \
        np.cos(np.radians(df['pickup_latitude'])) * \
        np.cos(np.radians(df['dropoff_latitude'])) * \
        (np.sin(alpha_longitude/2.0))**2

    c=2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
    df['distance']=R*c
    


def rush_hour_flag(df:pd.DataFrame):
    rush_hours={7,8,9,10,14,15,16,17}
    df['rush_hour']=df['hour'].isin(rush_hours)


def week_end_flag(df:pd.DataFrame):
    #the day is the a weekend if it is Saturday of Sunday which are mapped to 5,6
    df['weekend']=df['dayofweek'].isin([5,6])

def cyclical_features(df:pd.DataFrame):
    #we use the formulas:
        #x_sin = sin(2*pi*x/P)
        #x_cos = cos(2*pi*x/P)
    #where P is the maximum Known cycle
    df['hour_sin']=np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']=np.cos(2*np.pi*df['hour']/24.0)

    df['week_day_sin']=np.sin(2*np.pi*df['dayofweek']/7.0)
    df['week_day_cos']=np.cos(2*np.pi*df['dayofweek']/7.0)

def clean(df:pd.DataFrame):
    #standardizing our column names 
    df.columns=(
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ",'_')
    .str.replace('-','_')
    .str.replace('[^a-z0-9_]',"",regex=True))


    #capping the other features based on correct range for each feature
    df = df[df['trip_duration'].between(60, 10800)]
    df=df[df['passenger_count'].between(1,5)]

    df = df[
    (df['pickup_latitude'].between(40.4, 41.0))&
    (df['pickup_longitude'].between(-74.3, -73.6))&
    (df['dropoff_latitude'].between(40.4, 41.0))&
    (df['dropoff_longitude'].between(-74.3, -73.6))]

    return df.reset_index(drop=True)

def prepare_data(df:pd.DataFrame):
    df["pickup_datetime"]=pd.to_datetime(df['pickup_datetime'])
    #cleaning our noisy data
    df=clean(df)

    df['dayofweek']=df.pickup_datetime.dt.day_of_week
    # adding a feature to tell us if the day is a weekend
    week_end_flag(df)
    df['month']=df.pickup_datetime.dt.month
    df['hour']=df.pickup_datetime.dt.hour
    rush_hour_flag(df)
    df['trip_duration']=np.log1p(df['trip_duration'])
    #turing the hour of the day and day of the week from linear to cyclical
    cyclical_features(df)

    #getting the distance traveled each trip as a feature
    get_distance(df)

    #adding the bearing direction feature
    get_direction(df)

    #dropping un necessary columns
    df.drop(columns=['id','vendor_id','passenger_count','pickup_datetime'],inplace=True)
    return df

