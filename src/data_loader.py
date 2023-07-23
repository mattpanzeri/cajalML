import os, requests, tqdm
import numpy as np
import matplotlib.pyplot as plt

def retrieve_dataset():
    """
    download the dataset from the osf repository
    """
    fname = ['data/steinmetz_part%d.npz'%i for i in range(3)]
    url = ["https://osf.io/agvxh/download", 
           "https://osf.io/uv3mw/download", 
           "https://osf.io/ehmw2/download"]


    for j in tqdm.trange(len(url)):
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

def load_dataset(file, mouse):
    """
    loads the full dataset into memory
    """
    data = np.load('data/steinmetz_part%d.npz' % file, allow_pickle=True)['dat'][mouse]
    return data


if __name__ == "__main__":
    #retrieve_dataset()
    mouse_0 = load_dataset(0, 10)
    f, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].imshow(mouse_0['spks'][mouse_0['brain_area']=='MOp'][:,0], aspect='auto', cmap='gray_r')
    ax[1].plot(mouse_0['wheel'][0,0])
    plt.show()