import pytest
from pathlib import Path
import pandas as pd
from src.dashboard.app import DashboardApp

def test_dashboard_initialization():
    """Test dashboard initialization"""
    try:
        dashboard = DashboardApp()
        assert dashboard is not None
        print("✅ Dashboard initialization successful")
    except Exception as e:
        print(f"❌ Dashboard initialization failed: {e}")
        raise

def test_data_loading():
    """Test data loading functionality"""
    # Create test data if not exists
    data_file = Path('Primary_Parts_Code.csv')
    if not data_file.exists():
        df = pd.DataFrame({
            'Surveyor Part Code': ['P001', 'P002'],
            'Surveyor Part Name': ['Front Bumper', 'Rear Bumper']
        })
        df.to_csv(data_file, index=False)
    
    try:
        dashboard = DashboardApp()
        assert len(dashboard.data) > 0
        print("✅ Data loading successful")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        raise

if __name__ == "__main__":
    print("Running dashboard tests...")
    test_dashboard_initialization()
    test_data_loading()
    print("\nAll tests completed!") 