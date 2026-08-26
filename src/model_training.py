import xgboost as xgb 
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd 
from data_preprocessing import *
import os
import argparse


def model_eval(model,x,t,mesg='none'):
    pred=model.predict(x)
    mse=mean_squared_error(t,pred)
    r2=r2_score(t,pred)
    print(f'{mesg} evaluation')
    print(f"MSE: {mse}")
    print(f"R2 score: {r2}")
    print('================================================')


def test_model(test,model,features):
    model_eval(model,test[features],test.trip_duration,"test")

def train_model(train,val):
    
    log_features=['distance']#features the we will apply log transfomation on
    numeric_features = ['pickup_latitude','pickup_longitude',
                        'dropoff_longitude','dropoff_latitude','direction']#
    categorical_features = [ 'month','pickup_position'
                            ,'dropoff_position']#categoral data , we will apply one-hot encoding on it 
    no_scale=['week_day_sin','week_day_cos',
              'hour_sin','hour_cos','weekend','rush_hour']#features that does not need to be scaled

    #all of the features
    train_features = categorical_features + numeric_features+log_features+no_scale

    #calling the our coulmn tansfromer
    ct=scale(numeric_features,categorical_features,log_features)

    pipe=Pipeline(
        steps=[
        ('scale',ct),
         ('model',xgb.XGBRegressor())]
    )

    #train the model
    pipe.fit(train[train_features],train.trip_duration)

    #evaluate based on train set
    model_eval(pipe,train[train_features],train.trip_duration,'train')
    #evaluate based on validation set
    model_eval(pipe,val[train_features],val.trip_duration,'Validation')
    return pipe,train_features



def load_data(root_path):
    train=pd.read_csv(os.path.join(root_path,'train.csv'))
    val=pd.read_csv(os.path.join(root_path,'val.csv'))
    test=pd.read_csv(os.path.join(root_path,'test.csv'))

    return train,val,test



def main():
    parser=argparse.ArgumentParser(description="NYC trip duration predector")
    parser.add_argument('--root_path',type=str,default='/home/abdelrhman-elnaggar/ML/Projects/NYC-taxi-trip-duration/Data/split',help="the path for the data")
    args=parser.parse_args()
    root_path=args.root_path
    #load data
    train,val,test=load_data(root_path)

    #cleaning and adding our training features
    train=prepare_data(train)
    val=prepare_data(val)
    test=prepare_data(test)

    kmean_pickup,kmean_dropoff=positional_(train)
    val['pickup_position'],val['dropoff_position']=positional_(val,kmean_pickup,kmean_dropoff)
    test['pickup_position'],test['dropoff_position']=positional_(test,kmean_pickup,kmean_dropoff)

    #training the model
    pipe,features=train_model(train,val)
    #testing
    test_model(test,pipe,features)
    
    



if __name__=='__main__':
    main()
