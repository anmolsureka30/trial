import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_initial_data():
    """Create initial data files"""
    print("Creating test data...")
    
    # Create directories
    Path("data").mkdir(exist_ok=True)
    
    # Create Primary_Parts_Code.csv
    parts_data = pd.DataFrame({
        'Surveyor Part Code': [f'P{i:03d}' for i in range(100)],
        'Surveyor Part Name': [
            'Front Bumper', 'Rear Bumper', 'Hood', 'Windshield',
            'Headlight Left', 'Headlight Right', 'Door Front Left',
            'Door Front Right', 'Fender Left', 'Fender Right'
        ] * 10,
        'Category': ['Exterior', 'Exterior', 'Body', 'Glass', 
                    'Lighting', 'Lighting', 'Body', 'Body', 
                    'Exterior', 'Exterior'] * 10,
        'Average_Cost': np.random.uniform(500, 5000, 100),
        'Description': [
            'Front impact protection',
            'Rear impact protection',
            'Engine compartment cover',
            'Front window glass',
            'Left headlight assembly',
            'Right headlight assembly',
            'Left front door',
            'Right front door',
            'Left side panel',
            'Right side panel'
        ] * 10
    })
    
    # Save with consistent column names
    parts_data.to_csv('data/Primary_Parts_Code.csv', index=False)
    print("Created parts catalog")

    # Create historical_claims.csv with more realistic data
    num_claims = 1000
    start_date = datetime.now() - timedelta(days=365)
    
    claims_data = pd.DataFrame({
        'claim_id': [f'CLM{i:05d}' for i in range(num_claims)],
        'timestamp': pd.date_range(start=start_date, periods=num_claims),
        'total_amount': np.random.normal(5000, 1500, num_claims),
        'fraud_risk': np.random.beta(2, 10, num_claims) * 100,  # More realistic fraud risk distribution
        'processing_time': np.random.gamma(2, 2, num_claims),  # More realistic processing times
        'status': np.random.choice(
            ['approved', 'rejected', 'pending'],
            num_claims,
            p=[0.7, 0.2, 0.1]  # More realistic status distribution
        ),
        'vehicle_type': np.random.choice(
            ['Sedan', 'SUV', 'Truck', 'Van', 'Compact'],
            num_claims,
            p=[0.4, 0.3, 0.15, 0.1, 0.05]
        ),
        'damage_severity': np.random.choice(
            ['Minor', 'Moderate', 'Severe', 'Critical'],
            num_claims,
            p=[0.4, 0.3, 0.2, 0.1]
        ),
        'parts_affected': [
            np.random.choice(parts_data['Surveyor Part Name'].unique(), 
                           size=np.random.randint(1, 4)).tolist()
            for _ in range(num_claims)
        ],
        'recommendation': np.random.choice(
            ['approve', 'reject', 'review'],
            num_claims,
            p=[0.6, 0.2, 0.2]
        )
    })
    
    # Add some trends and patterns
    claims_data.loc[claims_data['total_amount'] > 10000, 'fraud_risk'] *= 1.5
    claims_data.loc[claims_data['processing_time'] > 5, 'status'] = 'pending'
    
    claims_data.to_csv('data/historical_claims.csv', index=False)
    print(f"Created {num_claims} historical claims")

    # Create garage data
    if not Path('data/garage.csv').exists():
        print("Creating garage data...")
        from src.data.create_garage_data import create_garage_data
        create_garage_data()

    # Create vectors directory
    Path('data/vectors/parts').mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_initial_data() 