import numpy as np
import pandas as pds
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from streamlit_option_menu import option_menu
from models.price import Base, Price
from datetime import datetime
import plotly.express as px


# -------------------------------------------------------------------------------ML
scaler_X = MinMaxScaler(feature_range=(0, 1))
# Configure the database
engine = create_engine('sqlite:///price.db')
Base.metadata.bind = engine

#@st.cache_data
def get_data():
    try :
        db_path = 'price.db'
        
        # Buat koneksi ke database menggunakan SQLAlchemy
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Baca seluruh data dari tabel ke dalam DataFrame
        query = 'SELECT * FROM price order by date desc' 
        df = pds.read_sql(query, engine)
        df = pds.DataFrame(df)
        
        return df
    except Exception as e:
        print(f"Error access db: {e}")

def normalized(sequence_data):
    middle_matrix = sequence_data[:, 1]
    print(middle_matrix)
    middle_matrix = middle_matrix.astype(float)
    print(middle_matrix)
    middle_matrix = middle_matrix.reshape(1, -1)
    print(middle_matrix)
    X_normalized = scaler_X.fit_transform(middle_matrix.reshape(-1, 1)).reshape(middle_matrix.shape + (1,))
    print(X_normalized)
    X_normalized = X_normalized.astype(np.float32)
    print(X_normalized)

    return X_normalized

def preprocess(days_difference):

    try:
        print("Fetching data...")
        df = get_data()
        print(df.head())
        print(type(df))
        print("Data fetched successfully.")

        print("Converting 'date' column to datetime...")
        df['date'] = pds.to_datetime(df['date'])
        print("'date' column converted successfully.")

        print("Converting numeric columns...")
        numeric_columns = ['close']
        df[numeric_columns] = df[numeric_columns].apply(pds.to_numeric, errors='coerce')
        print("Numeric columns converted successfully.")

        print("Checking for rows with null values...")
        df = df.fillna(df.mean())
        print("Rows with null values updated with mean.")

        print("Selecting columns...")
        df = df[['date', 'close']]
        print("Columns selected successfully.")

        print("Create sequence started...")
        df_sorted = df.sort_values(by='date', ascending=False)
        window = 3
        sequence_data = df_sorted.values[:window]
        print("Create sequence successfully.")

        print("Select last date started...")
        last_date = df['date'][:1]
        last_date = pds.Timestamp(last_date.iloc[0])
        print("Select last date successfully.")

        stat = 2

        return sequence_data, last_date, stat

    except Exception as e:
        print(f"Error in pre-process: {e}")
        return -2  # or return an error code or message

def next_weekday(d):
    while d.weekday() in {5, 6}:  # 5 adalah Sabtu, 6 adalah Minggu
        d += pds.DateOffset(days=1)
    return d


def denormalized(forecasted_values):
    reverse_forecast =  scaler_X.inverse_transform(forecasted_values.reshape(-1, 1)).flatten()
    return reverse_forecast


# Function to load LSTM model and make predictions
#@st.cache_data
def lstm_predict(num_predict):

    num_days_to_forecast = num_predict

    stat = 0 

    # Initialize an array to store the forecasted values
    forecasted_values = []

    try:
        loaded_model = load_model('models/model_lstm_s2.h5')
        stat = 1  # Update stat if the model is loaded successfully
        print("preprocess started...")
        sequence_data, last_date, stat = preprocess(num_days_to_forecast)
        print("preprocess ended.")
        print("normalization started...")
        X_normalized = normalized(sequence_data)
        print("normalization ended.")


        # Use the last few days from the test data to start the forecasting
        input_sequence = X_normalized

        for _ in range(num_days_to_forecast):
            # Make a prediction for the next day
            next_day_prediction = loaded_model.predict(input_sequence).flatten()[0]

            # Store the prediction in the forecasted_values array
            forecasted_values.append(next_day_prediction)

            # Update the input_sequence for the next prediction
            next_day_prediction = np.array([[next_day_prediction]])
            input_sequence = np.concatenate([input_sequence[:, 1:], np.expand_dims(next_day_prediction, axis=1)], axis=1)

        # Convert forecasted_values to a numpy array
        forecasted_values = np.array(forecasted_values)
        print(forecasted_values)

        # Create dates for the forecasted values
        forecast_dates = pds.date_range(start=last_date, periods=num_days_to_forecast + 1, freq='B').map(next_weekday)[1:]

        # Create a DataFrame to store the forecasted values and dates
        forecast_df = pds.DataFrame({
         'Date': forecast_dates,
            'Forecast': forecasted_values
        })

        # Display the forecast DataFrame
        print("forecast : ", forecast_df)
        print(forecast_df)

        reverse_forecast = denormalized(forecasted_values)

        reverse_forecast_df = pds.DataFrame({
            'Date': forecast_df['Date'].dt.strftime('%Y-%m-%d'),
            'Close': reverse_forecast
        })

        print("reverse_forecast_df: ",reverse_forecast_df)
        
        stat = 3

        print("end of process LSTM")
        return reverse_forecast_df

    except Exception as e:
        print(f"Error predicting using LSTM: {e}")
        stat = -2
        return forecast_df

def add_to_dataset(newData):
    try :
        db_path = 'price.db'
        
        # Buat koneksi ke database menggunakan SQLAlchemy
        engine = create_engine(f'sqlite:///{db_path}')
        
        session = Session(engine)

        # Baca seluruh data dari tabel ke dalam DataFrame
        isExist = pds.read_sql('SELECT * FROM price', engine)
        newData = newData[~newData['Date'].isin(isExist['Date'])]
    
        if not newData.empty:
            newData.to_sql('Price', con=engine, if_exists='append', index=False)
    except Exception as e:
        print(f"Error access db: {e}")

# -------------------------------------------------------------------------------UI
page_title = "OIL PRICE FORECAST APP"
page_icon = ":bar_chart:"
layout = "wide"

dataset = "https://finance.yahoo.com/quote/CL%3DF/history?p=CL%3DF"
sequenceLength = 3

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

sidebar = st.sidebar
with sidebar:
    ndays = st.number_input('Days to Predict:', min_value=1, max_value=90, value=10)
    st.markdown(f"Link to Dataset: {dataset}")
    #st.text_input('Window Slicing:',f"last {sequenceLength} days",disabled=True)
    submitted = st.button("Add to Dataset",type='primary')

st.title(":bar_chart: Forcasting Oil Price")
st.write("by Istifa Shania Putri(23523007) & Nadhira Virliany Daradjat(23523041)")

st.markdown("###")

left_column, right_column = st.columns(2)
with left_column:
    selected = option_menu(
        menu_title=None,
        options=["Data", "Visualization"],
        icons=["pencil-fill", "bar-chart-fill"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
)

# get historical data
hisData = get_data().rename(columns={'date': 'Date', 'close': 'Close'})
hisData.index = range(1, len(hisData) + 1)

# get forcast data
forcastData = lstm_predict(ndays)
forcastData.index = range(1, len(forcastData) + 1)

if selected == "Data":
    left_column, right_column, x_col = st.columns(3)
    with left_column:
        st.markdown("### Forcast Data")
        st.dataframe(forcastData)
    with right_column:
        st.markdown("### Historical Data")
        st.dataframe(hisData)   
     
if selected == "Visualization":
    left_column, right_column = st.columns(2)
    with left_column:
        fig = px.line(forcastData, x='Date', y='Close', title='Time Series of Forcast Price')
        st.plotly_chart(fig)
    with right_column:
        fig = px.line(hisData, x='Date', y='Close', title='Time Series of Historical Price')
        st.plotly_chart(fig)


if submitted:
    engine = create_engine('sqlite:///price.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    forcastData['Date'] = pds.to_datetime(forcastData['Date']).dt.date
    for index, row in forcastData.iterrows():
        exists = session.query(Price).filter_by(date=row['Date']).first()
        if not exists:
            new_price = Price(date=row['Date'], close=row['Close'])
            print(f"process date: {new_price.date}, close: {new_price.close}")
            session.add(new_price)
        else:
            print("skipped")

    session.commit()
    session.close()
    st.success('Data inserted successfully!')


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)