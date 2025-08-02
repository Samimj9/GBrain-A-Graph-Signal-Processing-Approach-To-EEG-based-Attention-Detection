import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
raw = mne.io.read_raw_fif(r'C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S01\S01_clean\Att_S01_raw.fif', preload=True)
clean = mne.io.read_raw_fif(r'C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S01\S01_clean\Att_S01_cleaned_raw2.fif', preload=True)
channels = clean.ch_names
data, times = raw.get_data(return_times=True)
clean_data, times = clean.get_data(return_times=True)
for i in range(len(channels)):
    raw = (data[i] - np.mean(data[i]))* 1e6    # Convert to µV* 1e6 
    clean = (clean_data[i] - np.mean(clean_data[i]))* 1e6  

    plt.figure(figsize=(12, 4))
    plt.plot(times, raw, label='Raw', alpha=0.4)
    plt.plot(times, clean, label='clean', alpha=0.8)
    plt.title(f"{channels[i]} - Raw vs clean")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.ylim(-100, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()