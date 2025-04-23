import pandas as pd
import  numpy as np
from datetime import datetime, timedelta
import argparse

class DataGenerator:
    def __init__(self):
        self.filters = {}

    def get_data(self, dataName='Data'):
        data  = self.data[dataName]      
    
        return data

class TestDataGenerator(DataGenerator):
    def __init__(self, dataName='TestData', num_cats=1, num_measures=1, num_personas=3, rows=100, include_timeseries=True):
        super().__init__()

        self.num_cats = num_cats
        self.num_measures = num_measures

        self.num_personas = num_personas
        self.num_rows = rows
        self.include_timeseries = include_timeseries

        self.data = { dataName:self._generate_data()
                     }

    def _generate_data(self):    
        data = {}


        # Optionally include a timeseries
        if self.include_timeseries:
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            interval = timedelta(days= 7)
            data['timestamp'] = [start_time + i%7 * interval for i in range(self.num_rows)]

        # Generate categorical columns with increasing cardinality
        for i in range(self.num_cats):
            cardinality = (self.num_personas)**(i+2) # cat_0 has 2 unique values, cat_1 has 3, etc.
            categories = [f"group_{j}" for j in range(cardinality)]
            data[f'dim_{i}'] = np.random.choice(categories, size=self.num_rows)


        # Generate measure columns
        for i in range(self.num_measures):
            data[f'measure_{i}'] = 50 + 10 * np.random.randn(self.num_rows) + np.random.choice([1, 5, 1000 ], p=[.5,.45 ,.05])

        return pd.DataFrame(data)

if __name__ == '__main__':
   #  python test/test_data_generator.py --size 100 --filename ./test/small-test.csv 
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', dest='size', type=int, help='Add number of rows to be generated')
    parser.add_argument('--personas', dest='personas', type=int, help='Add number of personas to be generated')
    parser.add_argument('--filename', dest='file', type=str, help='Add output file path')
    args = parser.parse_args()


    generator = TestDataGenerator(num_cats=2, num_measures=3, num_personas=10, rows=50, include_timeseries=True)
    df = generator.get_data('TestData')
    print(df.head())
    print("############################################################################################")
    generator = TestDataGenerator(num_cats=3, num_measures=3, num_personas=args.personas, rows=args.size, include_timeseries=True)
    df = generator.get_data('TestData')

    df.to_csv(args.file, index=False)

    print(df.head())