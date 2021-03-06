import os
import sys
import numpy as np

from common.io import mkpath,mkdir

types = ["100-100", "200-200", "400-400", "800-800", "100-100-100", "100-100-100-100", "100-100-100-100-100", '1600-1600'] 
large_types = ["800-800", "100-100-100-100-100", '1600-1600']
#types = ["100-100-100-100-100"] 
type_arch = {"25-25": "fcnn2", 
                "50-50": "fcnn2a", 
                "100-100": "fcnn2b", 
                "200-200": "fcnn2c", 
                "400-400": "fcnn2d", 
                "800-800": "fcnn2e", 
                "25-25-25": "fcnn3", 
                "50-50-50": "fcnn3a", 
                "100-100-100": "fcnn3b", 
                "25-25-25-25": "fcnn4", 
                "50-50-50-50": "fcnn4a", 
                "100-100-100-100": "fcnn4b", 
                "100-100-100-100-100": "fcnn5b",
                "1600-1600": "fcnn2f"}
c0 = np.arange(0,0.00021, 0.000025)
c1 = np.arange(0,0.00041, 0.000025)
l1_reg = { "25-25": [ 0.001 ], 
            "50-50": [ 0.0, 0.00015, 0.0003 ], 
            "100-100": c1,           
            "200-200": c0, 
            "400-400": c0,
            "800-800": c0, 
            "25-25-25": [0.0003], 
            "50-50-50": [0.0003], 
            "100-100-100": c0, 
            "25-25-25-25": [0.0007], 
            "50-50-50-50": [0.0, 0.0002, 0.0003],
            "100-100-100-100": c0, 
            "100-100-100-100-100": c0,
            "1600-1600": c0}
