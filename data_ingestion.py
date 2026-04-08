import pandas as pd
import os

def ingest_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    #Cleaning the Duplicates
    df.drop_duplicates(inplace=True)

    #make directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #Save cleaned csv
    df.to_csv(output_path,index=False)

    print(f'✅ Data injested and cleaned: {output_path}')
    return df