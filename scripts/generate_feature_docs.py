#!/usr/bin/env python
import sys, os, numpy as np, importlib
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from cnn_autoencoder import feature_extraction as fe
importlib.reload(fe)

fs = 1000.0
# create synthetic LFP (channels x time) and MUA
n_ch = 4
T = 5000
lfp = np.random.randn(n_ch, T) * 20e-6  # small volt-scale noise
mua = np.abs(np.random.randn(T))

# synthetic spectrogram: freq bins x time bins
spec = np.abs(np.random.randn(100, 50))
freqs = np.linspace(1, 300, 100)

# synthetic ripple_envelope full length
ripple_env = np.abs(np.random.randn(T))

# event dict
event = {
    'start_time': 0.5,
    'end_time': 0.6,
    'duration': 0.1,
    'peak_time': 0.55,
    'ripple_power': 1.23,
    'ripple_amplitude': 0.5,
    'spectrogram': spec,
    'spectrogram_freqs': freqs,
    'ripple_envelope': ripple_env,
    'peak_frequency': 150
}

features = fe.extract_swr_features(event, lfp, mua, fs, region_lfp={'CA1': lfp}, pre_ms=100, post_ms=100)
# features keys sorted as batch_extract_features would
feature_names = sorted(features.keys())

out_dir = os.path.join(project_root, 'cnn_autoencoder')
# write names to txt
with open(os.path.join(out_dir, 'events_feature_names.txt'), 'w') as f:
    for i, name in enumerate(feature_names):
        f.write(f"{i}\t{name}\n")

# write markdown description (brief)
md = '# SWR Feature Extraction — Feature List & Estimation Notes\n\n'
md += 'This file lists the exact feature names extracted per event and short notes on estimation.\\n\\n'
md += '## Feature names (ordered)\\n\\n'
for i, name in enumerate(feature_names):
    md += f"{i+1}. **{name}**\\n\\n"
md += '\n# End of file\n'
with open(os.path.join(out_dir, 'feature_descriptions.md'), 'w') as f:
    f.write(md)

print('WROTE', os.path.join(out_dir, 'events_feature_names.txt'))
print('WROTE', os.path.join(out_dir, 'feature_descriptions.md'))
print('\nFeature names:')
print(feature_names)
