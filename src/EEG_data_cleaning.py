import mne
from mne.preprocessing import ICA
import os # Import os module for path handling

# Define the path to your raw EEG file
# It's good practice to use os.path.join for cross-platform compatibility
raw_path = r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S03\S03_clean\Inatt_S03_raw.fif"

# Check if the file exists before attempting to load
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"The file specified does not exist: {raw_path}")

print(f"Loading raw EEG data from: {raw_path}")
raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False) # preload=True for filtering and ICA, verbose=False to reduce MNE output

print(f"Original channels: {raw.ch_names}")
print(f"Original number of channels: {len(raw.ch_names)}")

# --- Drop non-EEG channels (e.g., accelerometer channels if present) ---
# Enobio 8 typically provides 8 EEG channels and potentially accelerometer data (X, Y, Z).
# It's important to drop these if you only want to analyze EEG.
non_eeg_channels_to_drop = ['X', 'Y', 'Z'] # Add other non-EEG channels if you know them (e.g., ECG, EOG if separate)

# Filter out channels that don't exist in the raw data
channels_to_drop_exist = [ch for ch in non_eeg_channels_to_drop if ch in raw.ch_names]

if channels_to_drop_exist:
    raw.drop_channels(channels_to_drop_exist)
    print(f"Dropped non-EEG channels: {channels_to_drop_exist}")
else:
    print("No specified non-EEG channels ('X', 'Y', 'Z') were found to drop.")

# Check if there are still channels left, particularly EEG channels
if len(raw.ch_names) == 0:
    raise RuntimeError("No channels remaining after dropping. Cannot proceed with EEG analysis.")

print(f"Channels after dropping non-EEG: {raw.ch_names}")
print(f"Number of channels after dropping: {len(raw.ch_names)}")

# --- Set the standard 10-20 system montage ---
# MNE's 'standard_1020' montage is a common choice and contains locations
# for many standard EEG electrodes. Ensure your channel names match.
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    print("Montage 'standard_1020' set successfully for remaining channels.")

    # Optional: Verify that the montage was applied and check channel locations
    # raw.plot_sensors(show_names=True)

except ValueError as e:
    print(f"Could not set montage. Check remaining channel names in your raw data: {e}")
    print(f"Remaining channel names in your data: {raw.ch_names}")
    print("Ensure these names exactly match those in the 'standard_1020' montage (e.g., 'Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'Pz', 'O1', 'O2').")
    print("If names don't match, you may need to rename channels (e.g., raw.rename_channels({'OldName': 'NewName'})) or create a custom montage.")
    # You might want to exit or handle this error more gracefully depending on your pipeline requirements
    # For now, we'll proceed, but be aware that ICA plots might be inaccurate without correct locations.

# --- Filtering ---
# Apply bandpass filter (e.g., 1-40 Hz) to remove slow drifts and high-frequency noise.
# This is crucial before ICA.
print("Applying bandpass filter (1-40 Hz)...")
raw.filter(1, 40, fir_design='firwin', verbose=False)

# # Apply notch filter to remove power line noise (e.g., 50 Hz and 60 Hz)
# print("Applying notch filter (50 Hz and 60 Hz)...")
# raw.notch_filter(freqs=[50, 60], verbose=False)

# --- Re-referencing to Common Average Reference (CAR) ---
# This is a critical step for Enobio data, as the DRL/CMS provides a quasi-reference-free recording.
# CAR helps to approximate a neutral reference and is recommended before ICA.
# Ensure there's more than one channel to compute an average reference.
if len(raw.ch_names) > 1:
    print("Applying Common Average Reference (CAR)...")
    raw.set_eeg_reference(ref_channels='average', verbose=False)
    print("Common Average Reference applied.")
else:
    print("Not enough channels to apply Common Average Reference. Skipping re-referencing.")

# --- Independent Component Analysis (ICA) ---
# ICA is used to separate independent sources, including brain activity and artifacts (e.g., eye blinks, muscle activity).
# The number of components for ICA should be less than or equal to the number of channels.
# After applying CAR, the data rank is reduced by 1, so ICA can extract at most N-1 components.
n_channels_for_ica = len(raw.ch_names)
if n_channels_for_ica > 1: # Need at least 2 channels to perform ICA meaningfully (N-1 components)
    # n_components = min(8, len(raw.ch_names)) # Original line
    n_components = n_channels_for_ica - 1 # Corrected for CAR rank reduction
    
    print(f"Fitting ICA with {n_components} components (max possible after CAR).")
    ica = ICA(n_components=n_components, method='infomax', random_state=97, verbose=False)
    
    # Fit ICA to the raw data (which is now filtered and re-referenced)
    ica.fit(raw)
    print("ICA fitting complete.")

    # --- Plotting ICA results for inspection ---
    print("Plotting ICA components, sources, and properties for artifact identification.")
    print("Review these plots to identify and mark artifactual components (e.g., eye blinks, muscle artifacts).")
    print("You will then typically use ica.exclude to mark them for removal and ica.apply(raw) to clean the data.")

    # Plot component topographies (scalp maps)
    ica.plot_components(title='ICA Component Topographies')

    # Plot component time courses (activations)
    # Adjust the start and duration for better visualization if needed
    ica.plot_sources(raw, show_scrollbars=True, title='ICA Component Time Courses')

    # Plot properties of individual components (e.g., ERP image, spectrum)
    # This helps in identifying artifacts. You might need to specify which components to plot.
    # For now, let's plot a few components for example.
    try:
        # Plot properties for the first few components (e.g., up to 5)
        # Adjust 'picks' as needed to inspect specific components
        ica.plot_properties(raw, picks=range(min(n_components, 5)), title='ICA Component Properties') 
    except Exception as e:
        print(f"Could not plot ICA properties (might be due to data length or other issues): {e}")

else:
    print("Not enough channels remaining to perform ICA. Skipping ICA fitting and plotting.")

print("\nPreprocessing script finished. Remember to manually review ICA components to identify and exclude artifacts.")
print("After identifying artifact components, you would typically use: raw_cleaned = ica.apply(raw.copy(), exclude=ica.exclude)")

ica.plot_sources(raw, show_scrollbars=True, title='ICA Component Time Courses')
# After examining the plots, let's say you decide to exclude components with index 0 and 2
ica.exclude = [0,1,3]  # Example: Exclude components 0 and 2
# ica.exclude = [ 1,2,3,4,5,6,7]
print(f"Excluding components: {ica.exclude}")

# Apply the exclusion to the raw data
raw_cleaned = ica.apply(raw.copy())
print("ICA applied. Artifact components removed.")

# Plot a section of the cleaned data to visually inspect
# You might want to plot a segment where you know there were artifacts
raw_cleaned.plot(duration=150, start=0) # Adjust duration and start time as needed

# Define a path for the cleaned file
cleaned_raw_path = r"C:\Users\LENOVO\OneDrive - City University\Desktop\Attention detection FYP\S03\S03_clean\Inatt_S03_cleaned_raw3.fif"

# Save the cleaned raw data
raw_cleaned.save(cleaned_raw_path, overwrite=True)
print(f"Cleaned data saved to: {cleaned_raw_path}")