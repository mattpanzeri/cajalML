from tensorboard.backend.event_processing import event_accumulator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

def parse_tb(path, scalars):
    """
    parse the tensorboard file to extract the scalars
    """
    ea = event_accumulator.EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    data = {}
    for scalar in scalars:
        data[scalar] = np.array([scalar.value for scalar in ea.Scalars(scalar)])
    return data


#plt.figure()
#for p in Path(f'runs/session_{20}').glob('*'):
#    s = p.stem.split('_')
#    area, window, batch, epochs = s[0], s[2], s[4], s[6]
#    if (area=='MOp') and (window=='1'):
#        data = parse_tb(p.as_posix(), ["test/TN_mean", "test/AUC_std"])
#        mean = data["test/TN_mean"]
#        std = data["test/AUC_std"]
        
#        plt.plot(mean, label = "no_weight" if  "no_weight" in p.stem else "with_weight")
        #plt.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.5)
#plt.hlines(0.5, 0, 150, linestyle='--', color='k', label='chance')
#plt.xlabel('epochs')
#plt.ylabel('true negative rate')
#plt.xlim(0, 150)
#plt.legend()
#plt.tight_layout()
#plt.show()
f, ax = plt.subplots(1, 3, figsize=(15,5))
for i, session in enumerate([10, 20, 28]):
    aucs = {}
    for w in [1, 5, 10]:
        for p in Path(f'runs/session_{session}').glob(f'*_window_{w}_*'):
            s = p.stem.split('_')
            area, window, batch, epochs = s[0], s[2], s[4], s[6]
            if not "no_weight" in p.stem:
                data = parse_tb(p.as_posix(), ["test/AUC_mean"])
                mean = data["test/AUC_mean"][49]
                if area in aucs:
                    aucs[area].append(mean)
                else:
                    aucs[area] = [mean]

    
    for j, area in enumerate(aucs):
        ax[i].bar(np.arange(0,3) + j*3, aucs[area], color = ['b', 'g', 'r'])
    ax[i].set_xticks(ticks = np.arange(0, (j+1)*3, 3)+1, labels=aucs.keys())
            
            #if (window=='1') and (not ("no_weight" in p.stem)):
            #    data = parse_tb(p.as_posix(), ["test/AUC_mean", "test/AUC_std"])
            #    mean = data["test/AUC_mean"][49]
            #    if area in aucs:
            #        aucs[area].append(mean)
            #    else:
            #        aucs.append(mean)
            #    areas.append(area)

    ax[i].set_title(f"session {session}")
    ax[i].hlines(0.5, -0.5, (j+1)*3-0.5, linestyle='--', color='k', label='chance')
    ax[i].set_xlim(-0.5, (j+1)*3-0.5)
    ax[i].set_ylim(0, 1)

ax[0].set_ylabel("AUC score")
ax[1].set_xlabel("brain area")
plt.tight_layout()
plt.show()