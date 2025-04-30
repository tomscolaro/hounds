import pandas as pd
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shutil
import time
import numpy as np

class Track:
    def __init__(self, values:pd.DataFrame, dims,  log_directory, global_config = {'parent_':'Overall'}):
        self.global_config = global_config
        self.search_paths = values.copy()
        self.search_paths[ global_config['parent_']] =  global_config['parent_']
        self.search_paths = self.search_paths[[global_config['parent_']] + dims].reset_index().astype(str)

        self.dim = [global_config['parent_']] + dims
        
        self.dim_idx_map = {}        
        for i, _ in enumerate(self.dim):
            self.dim_idx_map[i] = self.dim[0:i+1]

        self.log_directory = log_directory      
        self.delete_all_files(self.log_directory)
        
        self.parent_track = global_config['parent_']

        self.track_manifest = [[self.parent_track]]
        self.max_level_idx = 0
        self.active_track = None

        self.anomaly_map = []

        sns.set_theme()
        return
    
    def write_anomaly_map(self):
        data = pd.DataFrame.from_records(self.anomaly_map)
        
        data = (
        data.groupby(self.dim, dropna=False)
            .agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
            .reset_index()
        )

        data.to_csv(self.log_directory+"/"+"manifest.csv", index=False)
        return
    
    def update_anomaly_map(self, dims, measures, active_track, res_obj,  active_measure):    
        temp_map = {}

        for idx, dim in enumerate(dims):
            if idx > len(active_track) - 1:
                temp_map[dim] = None
            else:        
                temp_map[dim] = active_track[idx]

        for idx, measure in enumerate(measures):
            if measure == active_measure:
                temp_map[measure + " Value Total"] = res_obj.observed.sum()
                temp_map[measure + " Value Average"] = res_obj.observed.mean()

                res_min = res_obj.resid.min()
                res_max = res_obj.resid.max()
                mag = res_max if np.abs(res_max) > np.abs(res_min) else res_min
                temp_map[measure + " Magnitude Residual Spike"] = mag

        self.anomaly_map.append(temp_map)
        return
        

    def log_track(self, values, ts, res, measure): 
        """
        this function takes these params:
            values: the current track. ie Overall/dim0/dim1
            ts: the aggregated data time series currently being analyzed
            res: the residual object created from the statsmodels STL fit object 
            measure: This is the measure currently being plotted

        this function plots to the log output directory and returns no objects 
            
        """
        # print(ts)

        directory = self.log_directory +"/"+"/".join(values) +"/"
        os.makedirs(directory, exist_ok=True)

        fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        axes = axes.flatten()  # easier to index
        d = {"Date": pd.to_datetime(ts), "Residuals": res.resid, "Seasonal":res.seasonal, "Trend": res.trend, "Series":res.observed}
        data = pd.DataFrame.from_dict(d)

        # print(data.head(10))

        # Plot each series in its own subplot
        for i, series in enumerate(["Series", "Trend", "Seasonal", "Residuals"]):
            ax = axes[i]
            sns.lineplot(data, x='Date', y=series, ax=ax, palette=['green'])
            ax.tick_params(axis='x', rotation=45)
            ax.set_title(series)
            ax.set_xlabel('Date')
            ax.set_ylabel(series)

            # Format x-axis dates
            # ax.xaxis.set_major_locator(mdates.MonthLocator())
            # Set x-axis ticks to show every 10th date
            ax.set_xticks(data['Date'][::5])
            ax.set_xticklabels(data['Date'][::5].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')

            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

                        # Add value labels every 5 points
            for _, row in data.iloc[::5].iterrows():
                ax.text(
                    row['Date'], row[series] +3,
                    f"{row[series]:.1f}",
                    fontsize=8, color='black',
                    ha='center', va='bottom'
                )
        
        # Adjust layout
        # plt.tight_layout()
        fig.savefig(directory+"{}chart.png".format(measure))
        plt.close('all')
        return

    def delete_all_files(self, dir_path):
        """Removes all files and subdirectories within a given directory.
        Args:
            dir_path: The path to the directory to be emptied.
        """
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return

        try:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path) # Recreate the directory
            print(f"All contents of '{dir_path}' have been removed.")
            time.sleep(3)
        except Exception as e:
            print(f"Error removing contents of '{dir_path}': {e}")

    def get_active_track(self):
        print("Track is currently {} items".format(len(self.track_manifest)))  
        if not self.track_manifest and self.max_level_idx < len(self.dim)-1:
            self.identify_track(self.active_track)
        if not self.track_manifest:
            return []
        active_track = self.track_manifest.pop(0)
        self.active_track = active_track
        return active_track

    def identify_track(self, values):
        dims = self.get_next_level_dims(values)
        if not dims or len(dims) == len(values):
            return        
        # print("log track dims", dims, "############################################")
        next_level_data =  self.filter_search_paths(values)[dims].drop_duplicates()
        # print("next level data shape: ", next_level_data.shape)
        for _, row in next_level_data.iterrows():
            self.track_manifest.append(row.tolist())
        # print("Track is now {} items".format(len(self.track_manifest)))
        return
    
    def get_next_level_dims(self, values):
        curr_max_idx = len(values) - 1      
        return  self.dim_idx_map.get(curr_max_idx+1, None)


    def filter_search_paths(self, values):
        conditions = {}
        for idx, val in enumerate(values):
            dim = self.dim[idx]
            conditions[dim] =  val

        mask = pd.Series(True, index=self.search_paths.index)
        for col, condition in conditions.items():
            mask &= self.search_paths[col] == condition

        return self.search_paths[mask]


    def get_data_filters(self, curr_level):
        conditions = {}
        relevant_dims = []
        track = []

        for idx, val in enumerate(curr_level):
            dim = self.dim[idx]
            relevant_dims.append(dim)
            track.append(val)
            conditions[dim] = val

        return conditions, relevant_dims, track


if __name__ == "__main__":
    # print("Release the hounds...")
    df = pd.read_csv('./test/test.csv')
    # print(df.shape)
    dims = ['dim_0', 'dim_1', 'dim_2']
    unqiue_values = df[dims].drop_duplicates().sort_values(dims)

    # print(df['timestamp'].unique())