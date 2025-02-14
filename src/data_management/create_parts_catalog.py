import pandas as pd
from pathlib import Path

def create_sample_catalog():
    """Create a sample parts catalog"""
    catalog_data = {
        'Part_Code': ['P001', 'P002', 'P003', 'P004'],
        'Standard_Name': ['Front Bumper', 'Rear Bumper', 'Hood', 'Front Door'],
        'Description': [
            'Front bumper assembly with fog lamp housing',
            'Rear bumper assembly with parking sensor holes',
            'Engine hood panel with insulation',
            'Front door assembly with window frame'
        ],
        'Category': ['Exterior', 'Exterior', 'Body', 'Body'],
        'Common_Variations': [
            'front bumper, bumper front, front bumper assy',
            'rear bumper, bumper rear, back bumper',
            'bonnet, engine cover, hood panel',
            'front door panel, door front, front door complete'
        ]
    }
    
    df = pd.DataFrame(catalog_data)
    output_path = Path('data/Primary_Parts_Code.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    create_sample_catalog() 