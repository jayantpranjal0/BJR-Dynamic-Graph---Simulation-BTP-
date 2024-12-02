import pandas as pd

def dataset_loader(csv, columns):
    df = pd.read_csv(csv)
    column_source_name = columns['source']
    column_destination_name = columns['destination']
    column_timestamp_name = columns['timestamp']
    
    no_of_timestamps = len(df[column_timestamp_name].unique())

    
    
    