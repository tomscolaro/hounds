import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL
from track import Track

class Hounds:
    def __init__(self,
                 data: pd.DataFrame,
                 dims:list, 
                 analysis_params:dict={
                                    "residual-confidence-threshold":3,
                },  
                 data_params:dict= {'time-series-column-name':"timestamp",  
                                    "aggs":{
                                        "measure_0": "sum",
                                        "measure_1":"sum", 
                                        "measure_2":"sum"
                                        }
                },
                track:Track=None):
        
        self.track = track
        self.analyis_params = analysis_params

        self.aggs = data_params['aggs']
        self.time_series_column = data_params['time-series-column-name']
        
        self.measures = self.define_measures(self.time_series_column, dims, data.columns)

        self.data =  data.sort_values([self.time_series_column] + dims,
                                       ascending=True).reset_index()
        #create an over-arching column to analzye all the data
        self.data[track.parent_track] = track.parent_track
        #re-order the data 
        self.data = self.data[[ self.time_series_column,track.parent_track] + dims + self.measures]
        # add the overarching category to the dims
        self.dims = [track.parent_track] + dims
            
    def run_hounds(self):
        print("Dims: {}  ||  Measures: {}".format(self.dims, self.measures))
        active_track = self.track.get_active_track()    

        while active_track:          
            agg_data, _, _ = self.filter_and_aggregate_data(active_track)
            # print("filtered and agg'd data shape {} for track {}".format(agg_data.shape, active_track))
            anomaly_detected = []    
            for measure in self.measures:
                residuals, res_obj =  self.decompose_series(agg_data[measure])
                anomaly_idx = self.detect_anomaly_from_residuals(residuals, \
                                                               self.analyis_params['residual-confidence-threshold'])
            
                if anomaly_idx:
                    anomaly_detected.append(measure)
                    self.track.log_track(active_track, res_obj, measure)

            if anomaly_detected:    
                self.track.identify_track(active_track)
                # print("######################################################")
            
            active_track = self.track.get_active_track()
        

        self.track.write_anomaly_map()
        return

    def decompose_series(self, data_series):
        stl = STL(data_series,6)
        res = stl.fit()
        return res.resid, res #get residual values from result object
    
    def detect_anomaly_from_residuals(self, res, thres):
        st_dev = res.std()
        return np.any(np.abs(res) > st_dev*thres)

    def filter_and_aggregate_data(self, active_track):
        query, relevant_dims, track = self.track.get_data_filters(active_track)

        filtered_data =  self.filter_data(query)
        agg_data = self.agg_level(filtered_data, relevant_dims)
        return agg_data, relevant_dims, track

    def filter_data(self, conditions):
        data_copy = self.data.copy()
        mask = pd.Series(True, index=data_copy.index)
        for col, condition in conditions.items():
            mask &= data_copy[col] == condition
        return data_copy[mask]
    def agg_level(self, data, relevant_dims):
        # print("Aggregating at Level: {}".format([self.time_series_column]+relevant_dims))
        return data.copy().groupby([self.time_series_column]+relevant_dims )\
                        .agg(self.aggs)\
                        .reset_index()

    def define_measures(self,ts_col, dims, data_cols):
        return  list(filter(lambda x: x not in ([ts_col] + dims), data_cols))

if __name__ == "__main__":
    df = pd.read_csv('./test/test.csv')
    print(df.shape)

    dims = ['dim_0', 'dim_1', 'dim_2']
    unqiue_values = df.copy()[dims].drop_duplicates() #.sort_values(dims)
    print("Search Path Size: {}".format(unqiue_values.shape[0]))

    tracks = Track(unqiue_values, dims, log_directory="log")

    hounds = Hounds(df, dims,   track=tracks)

    hounds.run_hounds()