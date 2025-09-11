import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure

def analyze_data(data_dict: dict) -> 'Figure': # using Figure only for annotation
    """
    pandas and matplotlib are only imported when this function is called.
    """
    print(f"pandas loaded before import: {'pandas' in sys.modules}")
    
    # import only when needed
    import pandas as pd
    import matplotlib.pyplot as plt
    
    print(f"pandas loaded after import: {'pandas' in sys.modules}")
    
    df = pd.DataFrame(data_dict)
    return df

def check_imports():
    heavy_modules = ['pandas', 'numpy', 'matplotlib']
    for module in heavy_modules:
        status = "LOADED" if module in sys.modules else "NOT LOADED"
        print(f"{module}: {status}")

if __name__ == "__main__":
    print("=== after module import ===")
    check_imports()
    
    print("\n=== after calling function ===")
    fig = analyze_data({'x': [1, 2, 3], 'y': [4, 5, 6]})
    check_imports()