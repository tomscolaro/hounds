import pandas as pd
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import shutil

class Track:
    def __init__(self, values:pd.DataFrame, dims,  log_directory, global_config = {'parent_':'Overall'}):
        self.global_config = global_config

        self.search_paths = values.copy()
        self.search_paths[ global_config['parent_']] =  global_config['parent_']
        self.search_paths = self.search_paths[[global_config['parent_']] + dims].reset_index()

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



        self.anomaly_map = {"Overall":[]}
        return
    

    def write_anomaly_map(self):
    
        return

    def log_track(self, values, res, measure): 
        
        # temp = []
        # for val in values:

        #     temp = self.anomaly_map[val]



        directory = self.log_directory +"/"+"/".join(values) +"/"
        # print(directory)
        os.makedirs(directory)
        # res.plot().cla()
        fig = res.plot()
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
        except Exception as e:
            print(f"Error removing contents of '{dir_path}': {e}")

    def get_active_track(self):
        print("Track is currently {} items".format(len(self.track_manifest)))  
        # print("Track contains {}".format(self.track_manifest))
        # print("Active Track is {}".format(active_track))                
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
            # if idx == 0:
            #     filter_str += " {} == '{}' ".format(dim, val)
            # else:
            #     filter_str += "& {} == '{}'".format(dim,val)
        # return filter_str, relevant_dims, track

        return conditions, relevant_dims, track


if __name__ == "__main__":
    # print("Release the hounds...")
    df = pd.read_csv('./test/test.csv')
    print(df.shape)

    dims = ['dim_0', 'dim_1', 'dim_2']
    unqiue_values = df[dims].drop_duplicates().sort_values(dims)

    # print("#########################################")
    # # print(unqiue_values)

    # print("#########################################")

    # tracks = Track(unqiue_values, dims, "log")

    # print(len(tracks.hierarchy))
    # print(tracks.filter_incrementer)

    # tracks.get_data_filters()
    # print(tracks.query_counter)
    print(df['timestamp'].unique())




