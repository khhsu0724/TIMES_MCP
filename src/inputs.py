import json
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, List

"""
    Author: Sean Hsu
    Date created: 9/17/2025
This file contains functions to generate input scripts for CTHFAM code
"""

### These Pydantic Classes Limit the behavior of LLM
class ControlParams(BaseModel):
    CF: List[float] = [0, 0, 1, 1, 1]          # Crystal field values
    SO: List[float] = [0, 0, 0]
    SC2: List[float] = [0, 0, 0, 0, 0]
    SC2EX: List[float] = [0, 0, 0, 0, 0]
    FG: List[float] = [0, 0, 0, 0]

class CellParams(BaseModel):
    Holes: int

class PhotonParams(BaseModel):
    XAS: bool = True
    RIXS: bool = False
    pvin: List[int] = [1, 1, 1]
    pvout: List[int] = [1, 1, 1]
    epsab: float = 0.5
    epsloss: float = 0.3

class InputParams(BaseModel):
    CONTROL: ControlParams = ControlParams()
    CELL: CellParams = CellParams(Holes=0)
    PHOTON: PhotonParams = PhotonParams()

def create_multiplet_input(input_params):
    """
    Generates an MCP INPUT file from a dictionary of parameters.

    Args:
        input_params (dict): A dictionary containing the parameters for the INPUT file,
                             organized by section (&CONTROL, &CELL, &PHOTON).

    Returns:
        str: The content of the MCP INPUT file.
    """

    # Default values for all parameters
    defaults = {
        "CONTROL": {
            "HFscale": 0.8,
            "DIAG": 4,
            "OVERWRITE": False,
            "EFFDEL": True,
            "CF": [0, 0, 1, 1, 1],
            "EXNEV": 10,
            "GSNEV": 10,
            # Below are options for hybridized system
            # "tpdzr": 1,
            # "MLCT": 16.8196,
            # "tpd": 1.1,
            # "tpp": 0,
            # "sigpi": 0.5,
        },
        "CELL": {
            "Coordination": "",
            "Sites": 1,
            "HYBMAT": "",
            "Holes": 0
        },
        "PHOTON": {
            "XAS": True,
            "RIXS": False,
            "pvin": [1, 1, 1],
            "pvout": [1, 1, 1],
            "solver": 4,
            "epsab": 0.5,
            "epsloss": 0.3,
            "niterCFE": 120,
            "CGTOL": 1e-8,
            "precond": 0,
            "NEDOS": 2000,
            "AB": [-20, 20],
            "ABMAX": 30, # Use ABMAX as default
            "INCIDENT": [8, 21, -1],
            "CROSS": False,
            "Edge": "L"
        }
    }

    output_lines = []

    for section in ["CONTROL", "CELL", "PHOTON"]:
        output_lines.append(f"&{section.upper()}")
        # section_params = input_params.get(section, {})
        
        # # Use provided parameters, falling back to defaults
        # all_params = {**defaults.get(section, {}), **section_params}

        section_model = getattr(input_params, section, None)
        section_dict = section_model.model_dump() if section_model else {}

        # Merge defaults with user-provided values
        all_params = {**defaults.get(section, {}), **section_dict}

        for key, value in all_params.items():
            # Format the value for the INPUT file
            if isinstance(value, list):
                formatted_value = " ".join(map(str, value))
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif isinstance(value, str) and value:
                if key == "Edge" or key == "EDGE": # Quirk of my code...
                    formatted_value = f'{value}'
                else:
                    formatted_value = f'"{value}"'
            elif value == "":
                formatted_value = f'""'
            else:
                formatted_value = str(value)

            output_lines.append(f"\t{key.upper()} = {formatted_value}")
        output_lines.append("/")
        output_lines.append("")

    return "\n".join(output_lines)

def get_dParams(element: str, valence: int):
    """
    Generate CONTROL section parameters for a given element and valence.

    Args:
        element (str): Element symbol (e.g. "Ni", "Cu").
        valence (int): Oxidation state (e.g. 2, 3).

    Returns:
        dict: Dictionary of CONTROL section values.
    """

    # Values taken from Haverkort's thesis. 
    # 3d SO for ground state is omitted for speed
    atomic_data = {
       "Zn": {
            3: {
                "SO": [15.738, 0.162, 0],
                "SC2": [4.5, 0.0, 14.489, 0.0, 9.041],
                "SC2EX": [4.5, 0.0, 15.210, 0.0, 9.495],
                "FG": [4.5, 7.084, 15.738, 4.033],
            },
            4: {
                "SO": [15.737, 0.175, 0],
                "SC2": [4.5, 0.0, 15.443, 0.0, 9.681],
                "SC2EX": [4.5, 0.0, 16.145, 0.0, 10.124],
                "FG": [4.5, 7.639, 15.737, 4.352],
            },
        },
        "Cu": {
            2: {
                "SO": [13.498, 0.124, 0],
                "SC2": [4.5, 0.0, 12.854, 0.0, 7.980],
                "SC2EX": [4.5, 0.0, 13.611, 0.0, 8.457],
                "FG": [4.5, 6.169, 13.498, 3.510],
            },
            3: {
                "SO": [13.496, 0.135, 0],
                "SC2": [4.5, 0.0, 13.885, 0.0, 8.669],
                "SC2EX": [4.5, 0.0, 14.617, 0.0, 9.130],
                "FG": [4.5, 6.708, 13.496, 3.818],
            },
            4: {
                "SO": [13.495, 0.147, 0],
                "SC2": [4.5, 0.0, 14.845, 0.0, 9.313],
                "SC2EX": [4.5, 0.0, 15.556, 0.0, 9.762],
                "FG": [4.5, 7.270, 13.495, 4.141],
            },
        },
        "Ni": {
            1: { 
                "SO": [11.509, 0.093, 0],               
                "SC2": [4.5, 0.0, 11.084, 0.0, 6.835],
                "SC2EX": [4.5, 0.0, 11.890, 0.0, 7.343],
                "FG": [4.5, 5.262, 7.098, 2.993],
            },
            2: {
                "SO": [11.507, 0.102, 0],
                "SC2": [4.5, 0.0, 12.233 , 0.0, 7.597],
                "SC2EX": [4.5, 0.0, 13.005, 0.0, 8.084],
                "FG": [4.5, 5.783, 7.720, 3.290],
            },
            3: {
                "SO": [11.506, 0.112, 0],
                "SC2": [4.5, 0.0, 13.276, 0.0, 8.294],
                "SC2EX": [4.5, 0.0, 14.021, 0.0, 8.763],
                "FG": [4.5, 6.329, 8.349, 3.602],
            },
            4: {
                "SO": [11.505, 0.122, 0],
                "SC2": [4.5, 0.0, 14.244, 0.0, 8.944],
                "SC2EX": [4.5, 0.0, 14.965, 0.0, 9.399],
                "FG": [4.5, 6.898, 8.984, 3.929],
            },
        },
        "Co": {
            0: {
                "SO": [9.752, 0.067, 0],
                "SC2": [4.5, 0.0, 2.196, 0.0, 1.348],
                "SC2EX": [4.5, 0.0, 9.969, 0.0, 6.108],
                "FG": [4.5, 4.367, 5.996, 2.482],
            },
            1: {
                "SO": [9.750, 0.075, 0],
                "SC2": [4.5, 0.0, 10.430, 0.0, 6.431],
                "SC2EX": [4.5, 0.0, 11.261, 0.0, 6.954],
                "FG": [4.5, 4.865, 6.624, 2.766],
            },
            2: {
                "SO": [9.748, 0.083, 0],
                "SC2": [4.5, 0.0, 11.604, 0.0, 7.209],
                "SC2EX": [4.5, 0.0, 12.395, 0.0, 7.707],
                "FG": [4.5, 5.394, 7.259, 3.068],
            },
            3: {
                "SO": [9.746, 0.092, 0],
                "SC2": [4.5, 0.0, 12.662, 0.0, 7.916],
                "SC2EX": [4.5, 0.0, 13.421, 0.0, 8.394],
                "FG": [4.5, 5.947, 7.899, 3.384],
            },
            4: {
                "SO": [9.746, 0.101, 0],
                "SC2": [4.5, 0.0, 13.638, 0.0, 8.572],
                "SC2EX": [4.5, 0.0, 14.372, 0.0, 9.034],
                "FG": [4.5, 6.525, 8.544, 3.716],
            },
        },
        "Fe": {
            0: {
                "SO": [8.203, 0.053, 0],
                "SC2": [4.5, 0.0, 2.087, 0.0, 1.275],
                "SC2EX": [4.5, 0.0, 9.293, 0.0, 5.688],
                "FG": [4.5, 3.957, 8.203, 2.248],
            },
            1: {
                "SO": [8.202, 0.059, 0],
                "SC2": [4.5, 0.0, 9.761, 0.0, 6.017],
                "SC2EX": [4.5, 0.0, 10.622, 0.0, 6.559],
                "FG": [4.5, 4.463, 8.202, 2.537],
            },
            2: {
                "SO": [8.200, 0.067, 0],
                "SC2": [4.5, 0.0, 10.965, 0.0, 6.815],
                "SC2EX": [4.5, 0.0, 11.778, 0.0, 7.327],
                "FG": [4.5, 5.000, 8.200, 2.843],
            },
            3: {
                "SO": [8.199, 0.074, 0],
                "SC2": [4.5, 0.0, 12.042, 0.0, 7.534],
                "SC2EX": [4.5, 0.0, 12.817, 0.0, 8.023],
                "FG": [4.5, 5.563, 8.199, 3.165],
            },
            4: {
                "SO": [8.199, 0.082, 0],
                "SC2": [4.5, 0.0, 13.029, 0.0, 8.197],
                "SC2EX": [4.5, 0.0, 13.775, 0.0, 8.667],
                "FG": [4.5, 6.150, 8.199, 3.502],
            },
        },
        "Mn": {
            0: {
                "SO": [6.849, 0.040, 0],
                "SC2": [4.5, 0.0, 5.312, 0.0, 3.268],
                "SC2EX": [4.5, 0.0, 8.594, 0.0, 5.255],
                "FG": [4.5, 3.539, 6.849, 2.010],
            },
            1: {
                "SO": [6.847, 0.046, 0],
                "SC2": [4.5, 0.0, 9.072, 0.0, 5.590],
                "SC2EX": [4.5, 0.0, 9.971, 0.0, 6.156],
                "FG": [4.5, 4.056, 6.847, 2.304],
            },
            2: {
                "SO": [6.846, 0.053, 0],
                "SC2": [4.5, 0.0, 10.315, 0.0, 6.413],
                "SC2EX": [4.5, 0.0, 11.154, 0.0, 6.942],
                "FG": [4.5, 4.603, 6.846, 2.617],
            },
            3: {
                "SO": [6.845, 0.059, 0],
                "SC2": [4.5, 0.0, 11.414, 0.0, 7.147],
                "SC2EX": [4.5, 0.0, 12.209, 0.0, 7.648],
                "FG": [4.5, 5.176, 6.845, 2.944],
            },
            4: {
                "SO": [6.845, 0.066, 0],
                "SC2": [4.5, 0.0, 12.414, 0.0, 7.819],
                "SC2EX": [4.5, 0.0, 13.176, 0.0, 8.299],
                "FG": [4.5, 5.773, 6.845, 3.287],
            },
        },
        "Cr": {
            0: {
                "SO": [5.671, 0.030, 0],
                "SC2": [4.5, 0.0, 6.833, 0.0, 4.155],
                "SC2EX": [4.5, 0.0, 7.866, 0.0, 4.803],
                "FG": [4.5, 3.112, 5.671, 1.767],
            },
            1: {
                "SO": [5.669, 0.035, 0],
                "SC2": [4.5, 0.0, 8.355, 0.0, 5.146],
                "SC2EX": [4.5, 0.0, 9.303, 0.0, 5.742],
                "FG": [4.5, 3.641, 5.669, 2.068],
            },
            2: {
                "SO": [5.668, 0.041, 0],
                "SC2": [4.5, 0.0, 9.648, 0.0, 6.001],
                "SC2EX": [4.5, 0.0, 10.521, 0.0, 6.551],
                "FG": [4.5, 4.201, 5.668, 2.387],
            },
            3: {
                "SO": [5.667, 0.047, 0],
                "SC2": [4.5, 0.0, 10.776, 0.0, 6.754],
                "SC2EX": [4.5, 0.0, 11.594, 0.0, 7.270],
                "FG": [4.5, 4.785, 5.667, 2.721],
            },
            4: {
                "SO": [5.668, 0.053, 0],
                "SC2": [4.5, 0.0, 11.793, 0.0, 7.437],
                "SC2EX": [4.5, 0.0, 12.572, 0.0, 7.928],
                "FG": [4.5, 5.394, 5.668, 3.070],
            },
        },
        "V": {
            0: {
                "SO": [4.652, 0.022, 0],
                "SC2": [4.5, 0.0, 5.970, 0.0, 3.620],
                "SC2EX": [4.5, 0.0, 7.095, 0.0, 4.324],
                "FG": [4.5, 2.673, 4.652, 1.517],
            },
            1: {
                "SO": [4.651, 0.026, 0],
                "SC2": [4.5, 0.0, 7.599, 0.0, 4.676],
                "SC2EX": [4.5, 0.0, 8.612, 0.0, 5.314],
                "FG": [4.5, 3.218, 4.651, 1.827],
            },
            2: {
                "SO": [4.650, 0.031, 0],
                "SC2": [4.5, 0.0, 8.961, 0.0, 5.576],
                "SC2EX": [4.5, 0.0, 9.875, 0.0, 6.152],
                "FG": [4.5, 3.792, 4.650, 2.154],
            },
            3: {
                "SO": [4.649, 0.036, 0],
                "SC2": [4.5, 0.0, 10.126, 0.0, 6.353],
                "SC2EX": [4.5, 0.0, 10.973, 0.0, 6.887],
                "FG": [4.5, 4.389, 4.649, 2.495],
            },
            4: {
                "SO": [4.650, 0.041, 0],
                "SC2": [4.5, 0.0, 0.000, 0.0, 0.000],
                "SC2EX": [4.5, 0.0, 11.963, 0.0, 7.554],
                "FG": [4.5, 5.011, 4.650, 2.852],
            },
        },
        "Ti": {
            0: {
                "SO": [3.778, 0.015, 0],
                "SC2": [4.5, 0.0, 5.002, 0.0, 3.019],
                "SC2EX": [4.5, 0.0, 6.261, 0.0, 3.806],
                "FG": [4.5, 2.216, 3.778, 1.257],
            },
            1: {
                "SO": [3.777, 0.019, 0],
                "SC2": [4.5, 0.0, 6.780, 0.0, 4.167],
                "SC2EX": [4.5, 0.0, 7.888, 0.0, 4.865],
                "FG": [4.5, 2.783, 3.777, 1.578],
            },
            2: {
                "SO": [3.776, 0.023, 0],
                "SC2": [4.5, 0.0, 8.243, 0.0, 5.132],
                "SC2EX": [4.5, 0.0, 9.213, 0.0, 5.744],
                "FG": [4.5, 3.376, 3.776, 1.917],
            },
            3: {
                "SO": [3.776, 0.027, 0],
                "SC2": [4.5, 0.0, 0.0, 0.0, 0.0],
                "SC2EX": [4.5, 0.0, 10.342, 0.0, 6.499],
                "FG": [4.5, 3.989, 3.776, 2.267],
            },
        },
        # Add more elements/valences here...
    }

    zero_val = {
        "Zn": -2,
        "Cu": -1,
        "Ni":  0,
        "Co":  1,
        "Fe":  2,
        "Mn":  3,
        "Cr":  4,
        "V":   5,
        "Ti":  6
    }
    holes = zero_val[element]+valence
    if holes < 0:
        raise ValueError(f"Invalid atomic configuration for {element} with valence {valence}")
    
    if element not in atomic_data or valence not in atomic_data[element]:
        raise ValueError(f"No atomic data found for {element} with valence {valence}")

    return holes, atomic_data[element][valence]
