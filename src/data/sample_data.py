import pandas as pd

def create_sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        'Surveyor Part Code': [f'P{i:03d}' for i in range(100)],
        'Surveyor Part Name': [
            'Front Bumper', 'Rear Bumper', 'Hood', 'Windshield',
            'Headlight Left', 'Headlight Right', 'Door Front Left',
            'Door Front Right', 'Fender Left', 'Fender Right'
        ] * 10
    }) 