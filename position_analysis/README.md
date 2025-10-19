# Position Analysis Utilities

This package centralizes loading, interpolation, kinematics, plotting, and linearization/HMM utilities for position data.

## Quick start

```python
from pfr_neurofunctions.position_analysis import (
    load_position_data,
    interpolate_position,
    estimate_velocity,
    plot_position_data,
    create_track_graph,
    linearize_positions_along_graph,
)

# Load
t, pos = load_position_data("/path/to/PositionData.mat", position_key="pos", return_time=True)

# Interpolate
lfp_time = ...  # your LFP time vector
_, pos_i = interpolate_position(t, pos.squeeze(), target_vector=lfp_time)

# Velocity
vel = estimate_velocity(t, pos.squeeze())

# Linearize along a simple track
G = create_track_graph({"A": (0,0), "B": (100,0)}, [("A", "B")])
lin, eids = linearize_positions_along_graph(G, xy_positions)
```
