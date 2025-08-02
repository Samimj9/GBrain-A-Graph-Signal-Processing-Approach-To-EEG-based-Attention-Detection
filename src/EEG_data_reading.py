import mne
import mne
import matplotlib.pyplot as plt
import numpy as np
# Example for EDF (adjust for your actual file format)
raw = mne.io.read_raw_edf(r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S02\S02_raw\20250513115446_Patient02_Protocol.edf", preload=True)
raw.info  # Print metadata

#Plotting the EEG data
raw.pick(['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'C3', 'Cz', 'C4'])
raw.plot(duration=10, scalings="auto")  # 10-second window

plt.style.use('default')
fig, ax = plt.subplots(figsize=(15, 10), facecolor='white')


# Extract and scale data
start, duration = 0, 5
data, times = raw[:, int(start*raw.info['sfreq']):int((start+duration)*raw.info['sfreq'])]

# Convert to microvolts (if not already)
try:
    if raw.info['chs'][0]['unit'] == 107:  # 107 is the code for volts in MNE
        data = data * 1e6  # Convert to µV
except:
    print("Assuming data is already in µV")

# Remove DC offset and scale properly
data = data - np.mean(data, axis=1, keepdims=True)

# Calculate proper spacing between channels
spacing = np.ptp(data) * 1.5  # Peak-to-peak range * 1.5

# Plot all channels in black with proper spacing
n_channels = min(20, len(raw.ch_names))
for i, ch_name in enumerate(raw.ch_names[:n_channels]):
    offset = i * spacing
    ax.plot(times, data[i] + offset, color='black', linewidth=1)
    # Add channel labels
    ax.text(times[0] - 0.1, offset, ch_name, 
            ha='right', va='center', color='black', fontsize=15)

# Customize the plot
ax.set_xlabel('Time (s)', color='black',fontsize=20)
ax.set_title('Raw EEG Signals', color='black',fontsize=20)

# Hide y-axis ticks since we're labeling channels directly
ax.set_yticks([])
ax.set_yticklabels([])

# Add grid lines
ax.grid(True, color='lightgray', linestyle=':', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Print all annotations (triggers)
print(raw.annotations)

for onset, duration, desc in zip(raw.annotations.onset,
                                 raw.annotations.duration,
                                 raw.annotations.description):
    print(f"Trigger '{desc}' starts at {onset:.2f} seconds and lasts {duration:.2f} seconds")

#Cropping the data to a specific time range
# Define epochs (e.g., -0.2s to 1.0s around each trigger)
epochs = mne.Epochs(
    raw,
    events=mne.events_from_annotations(raw)[0],  # Convert annotations to events
    event_id={"Trigger#1": 1, "Trigger#2": 2},   # Map descriptions to IDs
    tmin=-0.2,   # Start 200ms before trigger
    tmax=1.0,    # End 1s after trigger
    baseline=(None, 0),  # Baseline correct from tmin to 0
    preload=True,
)

attention_times = (100, 700)          # (start, end) in seconds
inattention_times = (1075, 1675)  # (start, end) in seconds

# Extract attention segment
raw_attention = raw.copy().crop(tmin=attention_times[0], tmax=attention_times[1])

# Extract inattention segment
raw_inattention = raw.copy().crop(tmin=inattention_times[0], tmax=inattention_times[1])

print(f"Attention duration: {raw_attention.times[-1]:.2f} sec")
print(f"Inattention duration: {raw_inattention.times[-1]:.2f} sec")

import mne
import os

# Load the raw EEG data
raw = mne.io.read_raw_edf(r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S04\20250519115355_Patient02_Protocol.edf", preload=True)

# Define attention and inattention times (start, end) in seconds
attention_times = (100, 700)  # Attention segment from 100s to 700s
inattention_times = (1075, 1675)  # Inattention segment from 1075s to 1675s

# Extract attention segment
raw_attention = raw.copy().crop(tmin=attention_times[0], tmax=attention_times[1])

# Extract inattention segment
raw_inattention = raw.copy().crop(tmin=inattention_times[0], tmax=inattention_times[1])

# Define file paths for saving in FIF format
att_file_path = r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\Att_S04_raw.fif"
inatt_file_path = r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\Inatt_S04_raw.fif"

# Check if the files already exist, and remove them if necessary
if os.path.exists(att_file_path):
    os.remove(att_file_path)
if os.path.exists(inatt_file_path):
    os.remove(inatt_file_path)

# Save the segments in FIF format
raw_attention.save(att_file_path)
raw_inattention.save(inatt_file_path)

print("Attention and Inattention segments have been saved successfully in FIF format.")
