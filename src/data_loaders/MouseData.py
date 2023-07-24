import os, requests, tqdm
import numpy as np
from pathlib import Path

class MouseData():
    def __init__(self, mouse):
        self.mouse = mouse
        self.load_data()
        self.recorded_brain_areas = np.unique(self.data['brain_area'])
        self.brain_areas = self.data['brain_area']
        self.spikes = self.data['spks']
        self.wheel = self.data['wheel']
        self.response = self.data['response']
        self.feedback_type = self.data['feedback_type']
        self.feedback_time = self.data['feedback_time']
        self.gocue = self.data['gocue']
        self.pupil = self.data['pupil']
        self.face = self.data['face']
        self.licks = self.data['licks']
        self.dt = self.data['bin_size']
        self.NN = self.spikes.shape[0]
        self.NTr = self.spikes.shape[1]
        self.NTi = self.spikes.shape[2]

    def load_data(self):
        """
        load the data from the npz file
        """
        file_id = self.mouse // 13
        fname = f'data/steinmetz_part{file_id}.npz'
        if not Path(fname).exists():    # download the dataset if it doesn't exist
            MouseData.retrieve_dataset()
        self.data = np.load(fname, allow_pickle=True)['dat'][self.mouse % 13]

    @staticmethod
    def retrieve_dataset():
        """
        download the dataset from the osf repository
        """
        fname = ['data/steinmetz_part%d.npz'%i for i in range(3)]
        url = ["https://osf.io/agvxh/download", 
               "https://osf.io/uv3mw/download", 
               "https://osf.io/ehmw2/download"]


        for j in tqdm.trange(len(url), desc='downloading data'):
            if not os.path.isfile(fname[j]):
                try:
                    r = requests.get(url[j])
                except requests.ConnectionError:
                    print("!!! Failed to download data !!!")
                else:
                    if r.status_code != requests.codes.ok:
                        print("!!! Failed to download data !!!")
                    else:
                        with open(fname[j], "wb") as fid:
                            fid.write(r.content)

if __name__ == "__main__":
    mouse = MouseData(38)
    print(mouse.recorded_brain_areas)