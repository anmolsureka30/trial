import pandas as pd
import os

def create_sample_data():
    """Create sample data for testing"""
    data = pd.DataFrame({
        'Surveyor Part Code': [f'P{i:03d}' for i in range(100)],
        'Surveyor Part Name': [
            'Front Bumper', 'Rear Bumper', 'Hood', 'Windshield',
            'Headlight Left', 'Headlight Right', 'Door Front Left',
            'Door Front Right', 'Fender Left', 'Fender Right'
        ] * 10
    })
    
    # Save to CSV
    data.to_csv('Primary_Parts_Code.csv', index=False)
    return data

if __name__ == "__main__":
    create_sample_data() 