import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, PowerTransformer

from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from datetime import timedelta
from train_model import train


def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv', delimiter = ',')
    df.to_csv("cars.csv", index=False)
    print("df: ", df.shape)

    return df


def clear_data():
    df = pd.read_csv("cars.csv")
    
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    
    question_dist = df[(df.Year < 2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)
    # Анализ и очистка данных
    # анализ гистограмм
    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)
    
    # здравый смысл
    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)
    
    # анализ гистограмм
    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)
    
    # анализ гистограмм
    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)
    
    df = df.reset_index(drop=True)  
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    df.to_csv('df_clear.csv')

    return True


dag_cars = DAG(
    dag_id="train_pipe",
    start_date=datetime(2025, 5, 26),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)


download_task = PythonOperator(python_callable=download_data, task_id="download_cars", dag=dag_cars)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_cars", dag=dag_cars)
train_task = PythonOperator(python_callable=train, task_id="train_cars", dag=dag_cars)

download_task >> clear_task >> train_task
