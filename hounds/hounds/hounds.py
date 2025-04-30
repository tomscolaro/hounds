import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from track import Track
import argparse
import json

class Hounds:
    def __init__(self,
                 data: pd.DataFrame,
                 dims:list, 
                 analysis_params,
                 data_params,
                track:Track=None):
        
        self.track = track
        self.analyis_params = analysis_params

        # self.aggs = data_params['aggs']
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
                
                #if the average of a measure falls below our anomaly floor, don't even run the anomaly detection on it
                if agg_data[measure].mean() < self.analyis_params['anomaly-floor'][measure] or agg_data.shape[0] <= 1:
                    continue

                res_obj =  self.decompose_series(agg_data[measure])
                anomaly_idx = self.detect_anomaly_from_residuals(res_obj.resid, \
                                                               self.analyis_params['residual-confidence-threshold'])

                # print(np.max(np.where(anomaly_idx)[0]))
                indices  = np.where(anomaly_idx)[0]
                if indices.size == 0:
                    continue
    
                periods_behind = agg_data.shape[0]  - np.max(indices)
  
                if periods_behind < self.analyis_params['lookback-limit']:
                    anomaly_detected.append(measure)
                    self.track.log_track(active_track, agg_data[self.time_series_column], res_obj, measure)
                    self.track.update_anomaly_map(self.dims, self.measures, active_track, res_obj, measure)

            if anomaly_detected:    
                self.track.identify_track(active_track) # reinserts tracks of interest
        
            active_track = self.track.get_active_track()
        

        self.track.write_anomaly_map()
        return

    def decompose_series(self, data_series):
        stl = STL(data_series,self.analyis_params['stl-periods'],  robust=self.analyis_params['robust'])
        res = stl.fit()
        return res #get residual values from result object
    
    def detect_anomaly_from_residuals(self, res, thres):
        st_dev = res.std()
        match self.analyis_params['analysis-type']:
            case "positive":
                return res > st_dev*thres
            case "negative":
                return res < -1 * st_dev*thres
            case _:
                return np.abs(res) > st_dev*thres

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
        return data_copy[mask].reset_index()
    
    def agg_level(self, data, relevant_dims):
        return data.copy().groupby([self.time_series_column]+relevant_dims )\
                        .sum()\
                        .reset_index()

    def define_measures(self,ts_col, dims, data_cols):
        return  list(filter(lambda x: x not in ([ts_col] + dims), data_cols))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file', type=str, help='Add File path')
    parser.add_argument('--dims', dest='dims', nargs='+', help='Add Dim List')
    parser.add_argument('--output_path', dest='output_path', type=str, help='Add output file path') 
    parser.add_argument('--timestamp-col', dest='timestamp_col', default='timestamp', type=str)  
    parser.add_argument('--n-period-lookback-limit', dest='lookback_limit', default=100, type=int)
    parser.add_argument('--resid-stdev-thres', dest='resid_thres', default=3, type=float)
    parser.add_argument('--stl-periods', dest='stl_periods', default=12, type=int)
    parser.add_argument('--anomaly-floor', dest='anomaly_floor', type=json.loads, help='Dictionary in JSON format')
    parser.add_argument('--robust', dest='robust', type=bool, default=True, help='Dictionary in JSON format')
    parser.add_argument('--analysis-type', dest='analysis_type', type=str, default='both', help='Dictionary in JSON format')
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    print(df.shape, df.columns)
  


    dims = args.dims
    analysis_params = {
        "residual-confidence-threshold":args.resid_thres,
        "lookback-limit": args.lookback_limit,
        "stl-periods": args.stl_periods,
        'anomaly-floor': args.anomaly_floor,
        "robust": args.robust,
        'analysis-type': args.analysis_type
    }  
    
    data_params = {
        'time-series-column-name':args.timestamp_col,  
    }

    unqiue_values = df.copy()[dims].drop_duplicates() #.sort_values(dims)


    print("Search Path Size: {}".format(unqiue_values.shape[0]))
    tracks = Track(unqiue_values, dims, log_directory=args.output_path)
    hounds = Hounds(df, dims,  analysis_params, data_params,  track=tracks)
    hounds.run_hounds()

