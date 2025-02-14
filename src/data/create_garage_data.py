import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_garage_data():
    """Create sample garage data with variations in part names"""
    # Load standard parts data
    standard_parts = pd.read_csv('data/Primary_Parts_Code.csv')
    
    # Define common variations and typos with more structured mapping
    variations = {
        'Front Bumper': {
            'variations': ['Front Bumper Assembly', 'Frt Bumper', 'Bumper-Front', 'Front Bumper Assy'],
            'common_errors': ['Front Buffer', 'F Bumper', 'Bumper F']
        },
        'Rear Bumper': {
            'variations': ['Back Bumper', 'Rear Bumper Assy', 'Bumper-Rear', 'Rear Bumper Assembly'],
            'common_errors': ['Back Buffer', 'R Bumper', 'Bumper R']
        },
        'Hood': {
            'variations': ['Bonnet', 'Engine Hood', 'Front Hood', 'Hood Panel'],
            'common_errors': ['Bonit', 'Engine Cover', 'Front Cover']
        },
        'Windshield': {
            'variations': ['Front Glass', 'Windscreen', 'Front Windshield', 'Wind Shield'],
            'common_errors': ['Front Window', 'Windshild', 'Shield Glass']
        },
        'Headlight': {
            'variations': ['Head Lamp', 'Front Light', 'Head Light Assembly', 'Front Headlamp'],
            'common_errors': ['Head Lite', 'Front Lamp', 'Head Light']
        }
    }

    # Create garage entries with more detailed mapping information
    garage_entries = []
    num_entries = 1000
    start_date = datetime.now() - timedelta(days=90)

    for i in range(num_entries):
        # Select a random standard part
        std_part = standard_parts.sample(1).iloc[0]
        base_name = std_part['Surveyor Part Name']
        
        # Generate variation or error
        use_error = random.random() < 0.15  # 15% chance of error
        for key, var_data in variations.items():
            if key in base_name:
                if use_error:
                    part_desc = random.choice(var_data['common_errors'])
                    confidence = random.uniform(30, 60)
                    error_type = 'Incorrect Naming'
                else:
                    part_desc = random.choice(var_data['variations'])
                    confidence = random.uniform(70, 100)
                    error_type = None
                break
        else:
            part_desc = base_name
            confidence = random.uniform(80, 100)
            error_type = None

        entry = {
            'garage_id': f'G{i:04d}',
            'part_description': part_desc,
            'mapped_part_code': std_part['Surveyor Part Code'] if confidence > 50 else None,
            'mapped_part_name': std_part['Surveyor Part Name'] if confidence > 50 else None,
            'mapping_confidence': confidence,
            'part_category': std_part['Category'],
            'error_type': error_type,
            'mapping_timestamp': (start_date + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )).strftime('%Y-%m-%d %H:%M:%S'),
            'is_error': use_error,
            'mapping_source': 'AI Model',
            'processing_time': random.uniform(0.5, 5.0)
        }
        garage_entries.append(entry)

    # Create DataFrame and save
    garage_df = pd.DataFrame(garage_entries)
    garage_df.to_csv('data/garage.csv', index=False)
    print(f"Created {len(garage_df)} garage entries")

if __name__ == "__main__":
    create_garage_data() 