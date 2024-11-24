import pandas as pd
import numpy as np 


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  GRU, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping


from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import os

import warnings
def handle(name):
    warnings.simplefilter(action="ignore")

    file_path=name
    data = pd.read_csv(f"csv_files\{file_path}")

    data

    data["Date"] = data.loc[:,"Date"].str.extract("(^.{10})")

    data["Date"] = pd.to_datetime(data.loc[:,"Date"])

    data = data.set_index("Date")

    dataset = data[['Open','Close','High','Low']]
    model_data = pd.DataFrame(dataset)

    model_data = model_data.values

    scaler = MinMaxScaler()
    model_data = scaler.fit_transform(model_data)

    timestep = 60

    train_size = int(len(model_data)*.8)
    test_size = len(data)-train_size

    train_data = model_data[:train_size,:]
    test_data = model_data[train_size-timestep:,:]

    x_train = []
    y_train = []

    for i in range(timestep, len(train_data)):
        x_train.append(train_data[i-timestep:i, :])
        y_train.append(train_data[i, :])
        
        
    x_train,y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 4))

    x_test = []
    y_test = []

    for i in range(timestep, len(test_data)):
        x_test.append(test_data[i-timestep:i,:])
        y_test.append(test_data[i,:])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))

    model = Sequential([
        GRU(150, return_sequences= False, input_shape= (x_train.shape[1], 4)),
        Dropout(0.4),
        Dense(64),
        Dropout(0.2),
        Dense(4)
    ])

    model.compile(optimizer= 'adam', loss= 'mse' , metrics= ['mean_absolute_error','r2_score','root_mean_squared_error'])

    callbacks = [EarlyStopping(monitor= 'loss', patience= 10 , restore_best_weights= True)]
    history = model.fit(x_train, y_train, epochs= 25, batch_size= 16 , callbacks= callbacks )

    df = pd.DataFrame(history.history)
    df = df.rename(columns={"loss":"mean_squared_error"})

    df = pd.DataFrame(history.history)
    df = df.rename(columns={"loss":"mean_squared_error"})

    model.evaluate(x_test,y_test)

    predictions = model.predict(x_test)

    #inverse predictions scaling
    predictions = scaler.inverse_transform(predictions)

    y_test = scaler.inverse_transform(y_test)

    RMSE = np.sqrt(np.mean( y_test - predictions )**2).round(2)


    train = pd.DataFrame(dataset.iloc[:train_size ])
    test = pd.DataFrame(dataset.iloc[train_size:])



    from datetime import timedelta

    def insert_end(Xin, new_input):
        for i in range(timestep - 1):
            Xin[:, i, :] = Xin[:, i+1, :]
        Xin[:, timestep - 1, :] = new_input
        return Xin

    future = 30
    forcast = []
    Xin = x_test[-1 :, :, :]
    time = []
    for i in range(0, future):
        out = model.predict(Xin, batch_size=5)
        forcast.append(out[0, :]) 
        print(forcast)
        Xin = insert_end(Xin, out[0, 0]) 
        time.append(pd.to_datetime(data.index[-1]) + timedelta(days=i))

    df = pd.concat([train,test.iloc[:,:-1]],axis=0)

    forcasted_output = np.asanyarray(forcast)   
    forcasted_output = forcasted_output.reshape(-1, 4) 
    forcasted_output = scaler.inverse_transform(forcasted_output) 

    forcasted_output = pd.DataFrame(forcasted_output)
    date = pd.DataFrame(time)
    df_result = pd.concat([date,forcasted_output], axis=1)
    df_result.columns = "date", "forecasted_open","forecasted_close","forecasted_high","forecasted_low"
    CSV_FOLDER = os.path.join(os.getcwd(), "csv_files")
    if not os.path.exists(CSV_FOLDER):
        os.makedirs(CSV_FOLDER)

    csv_file_name2= f"{name}_PRED_data.csv"
    csv_file_path2= os.path.join(CSV_FOLDER, csv_file_name2)
    df_result.head(1)
    df_result.to_csv(csv_file_path2,index=False)

