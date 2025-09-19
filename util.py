import re

def extract_occupation(filename):
    """
    Reads a file and extracts 'Num Holes' and orbital occupations.
    Returns a dictionary:
    {
        'Num Holes': <int>,
        'orbitals': {'dx2': 0.99647, 'dz2': 0.99647, ...}
    }
    """
    result = {}
    orbitals = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Find the line with "Num Holes"
    for i, line in enumerate(lines):
        if "Num Holes" in line:
            result["Num Holes"] = int(line.split(":")[1].strip())
            j = i + 1
            while j < len(lines):
                l = lines[j].strip()
                if l[0] ==  "-":
                    break  # stop at line break
                parts = l.split()
                if len(parts) == 2:
                    orb, val = parts
                    orbitals[orb] = float(val)
                j += 1
            break
    
    result["orbitals"] = orbitals
    return result

def extract_ground_state(filename):
    """
    Reads a file and extracts the 'Ground State composition' line.
    Returns a dictionary with states as keys and values as floats.
    """
    with open(filename, "r") as f:
        for line in f:
            if "Ground State composition" in line:
                _, rest = line.split(",", 1)
                # Split by comma and parse each state
                state_dict = {}
                for item in rest.split(","):
                    key, val = item.split(":")
                    state_dict[key.strip()] = float(val.strip())
                return state_dict
    return {}

# Example usage:
filename = "/Users/seanhsu/Desktop/School/Research/Program File/ED/mcpruns/b95572a68fe9/eig.txt"
data = extract_occupation(filename)
print(data)

filename = "/Users/seanhsu/Desktop/School/Research/Program File/ED/mcpruns/b95572a68fe9/ed.out"
data = extract_ground_state(filename)
print(data)