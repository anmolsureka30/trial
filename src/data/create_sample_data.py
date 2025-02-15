import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data(num_claims: int = 1000) -> pd.DataFrame:
    """Create sample claims data for testing"""
    
    # Generate random claim IDs
    claim_ids = np.arange(1, num_claims + 1)
    
    # Generate random part codes
    part_codes = [f"{x:04d}" for x in range(1, 21)]
    
    # Generate sample data
    data = {
        'claim_id': np.repeat(claim_ids, 3),  # Each claim has ~3 damaged parts
        'damaged_part_code': np.random.choice(part_codes, size=num_claims * 3),
        'damage_severity': np.random.choice(
            ['minor', 'moderate', 'severe', 'critical'],
            size=num_claims * 3,
            p=[0.4, 0.3, 0.2, 0.1]
        )
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    create_sample_data() 