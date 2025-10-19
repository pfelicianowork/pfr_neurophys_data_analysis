import os
from typing import Tuple, Union, Optional, List, Dict
import numpy as np
import scipy.io


def load_all_position_data(
    file_path: str,
    struct_name: Optional[str] = None,
    fields: Optional[List[str]] = None,
    maintain_2d: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Flexible loader for position and other data from MATLAB .mat files.
    Returns a dict of numpy arrays with consistent lengths.
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True)

    # detect the main struct
    if struct_name is None:
        possible_structs = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(possible_structs) == 0:
            raise ValueError("No valid structs found in .mat file")
        struct_name = possible_structs[0]

    if struct_name not in mat_data:
        raise KeyError(f"Struct '{struct_name}' not found in .mat file")

    data_struct = mat_data[struct_name]
    available_fields = getattr(data_struct, "dtype", None)
    if available_fields is None or data_struct.dtype.names is None:
        raise ValueError("No fields found in the data struct")
    available_fields = data_struct.dtype.names

    if fields is None:
        fields = list(available_fields)

    if maintain_2d is None:
        maintain_2d = ['pos', 'position']

    output_data: Dict[str, np.ndarray] = {}
    for field in fields:
        if field not in available_fields:
            raise KeyError(f"Field '{field}' not found. Available fields: {available_fields}")

        arr = np.asarray(data_struct[field].item())
        if field in maintain_2d:
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim == 2:
                pass
            else:
                raise ValueError(f"Field '{field}' has unexpected dimensions: {arr.ndim}")
        else:
            if arr.ndim > 1:
                arr = arr.flatten()
        output_data[field] = arr

    lengths = [a.shape[0] for a in output_data.values()]
    if not all(l == lengths[0] for l in lengths):
        raise ValueError("Mismatch in data lengths across fields")
    return output_data


def load_position_data(
    file_path: str,
    struct_name: Optional[str] = None,
    time_key: Optional[str] = None,
    position_key: Optional[str] = None,
    return_time: bool = True,
) -> Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]]:
    """
    Flexible loader for time and position (1D: linear, 2D: x,y) from MATLAB .mat files.
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True)

    if struct_name is None:
        possible_structs = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(possible_structs) == 0:
            raise ValueError("No valid structs found in .mat file")
        struct_name = possible_structs[0]

    if struct_name not in mat_data:
        raise KeyError(f"Struct '{struct_name}' not found in .mat file")
    data_struct = mat_data[struct_name]

    available_fields = getattr(data_struct, "dtype", None)
    if available_fields is None or data_struct.dtype.names is None:
        raise ValueError("No fields found in the data struct")
    available_fields = data_struct.dtype.names

    # position
    position_candidates = ['pos', 'position', 'linearpos', 'linear_pos', 'linVel'] if position_key is None else [position_key]
    position_fields = [f for f in position_candidates if f in available_fields]
    if not position_fields:
        raise ValueError(f"No position fields found. Available fields: {available_fields}")

    position_data: List[np.ndarray] = []
    for field in position_fields:
        pos = np.asarray(data_struct[field].item())
        if field in ['pos', 'position']:
            if pos.ndim == 1:
                pos = pos.reshape(-1, 1)
            elif pos.ndim == 2 and pos.shape[1] >= 2:
                pos = pos[:, :2]
        else:
            if pos.ndim > 1:
                pos = pos.flatten()
        position_data.append(pos)

    if len(position_data) == 1:
        position_data = position_data[0]  # type: ignore

    if not return_time:
        return position_data  # type: ignore

    # time
    time_candidates = ['time', 't', 'timestamp'] if time_key is None else [time_key]
    time_field = next((c for c in time_candidates if c in available_fields), None)
    if time_field is None:
        raise ValueError(f"No time field found. Available fields: {available_fields}")

    time_data = np.asarray(data_struct[time_field].item())

    # length check
    if isinstance(position_data, list):
        for pos in position_data:
            if pos.shape[0] != len(time_data):
                raise ValueError(f"Mismatch between time ({len(time_data)}) and position ({pos.shape[0]})")
    else:
        if position_data.shape[0] != len(time_data):  # type: ignore[union-attr]
            raise ValueError(f"Mismatch between time ({len(time_data)}) and position ({position_data.shape[0]})")  # type: ignore[index]

    return time_data, position_data  # type: ignore[return-value]


def get_file_nested_keys(file_path: str) -> Tuple[str, List[str]]:
    """
    Find the struct name and extract all nested keys from a MATLAB .mat file.
    """
    mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    def _extract_nested_keys(mat_struct, parent_key: str = "") -> List[str]:
        keys: List[str] = []

        # old MATLAB struct type
        try:
            from scipy.io.matlab.mio5_params import mat_struct
            is_mat_struct = isinstance(mat_struct, mat_struct)
        except Exception:
            is_mat_struct = False

        if is_mat_struct:
            for field in mat_struct._fieldnames:
                full_key = f"{parent_key}.{field}" if parent_key else field
                keys.append(full_key)
                sub_struct = getattr(mat_struct, field)
                keys.extend(_extract_nested_keys(sub_struct, full_key))

        elif isinstance(mat_struct, dict):
            for field, sub_struct in mat_struct.items():
                if not str(field).startswith('__'):
                    full_key = f"{parent_key}.{field}" if parent_key else str(field)
                    keys.append(full_key)
                    keys.extend(_extract_nested_keys(sub_struct, full_key))

        elif isinstance(mat_struct, np.ndarray):
            if getattr(mat_struct, "dtype", None) is not None and mat_struct.dtype.names:
                for field in mat_struct.dtype.names:
                    full_key = f"{parent_key}.{field}" if parent_key else field
                    keys.append(full_key)
                    sub_struct = mat_struct[field]
                    keys.extend(_extract_nested_keys(sub_struct, full_key))
            elif mat_struct.size > 0 and hasattr(mat_struct.item(0), '_fieldnames'):
                first_item = mat_struct.item(0)
                keys.extend(_extract_nested_keys(first_item, parent_key))

        return keys

    struct_names = [k for k in mat_data.keys() if not k.startswith('__')]
    if not struct_names:
        raise ValueError("No valid structs found in the .mat file")

    struct_name = struct_names[0]
    nested_keys = _extract_nested_keys(mat_data[struct_name])
    return struct_name, nested_keys
