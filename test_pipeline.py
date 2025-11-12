# test_pipeline.py
"""
Test script to verify the pipeline works
"""
import sys
import os

# Add src to Python path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported"""
    try:
        from src.config import DATA_PATH, RANDOM_STATE
        from src.data_loader import load_data
        from src.preprocessing import engineer_features
        from src.model_training import compare_models
        from src.pipeline import run_full_pipeline
        
        print(" All imports successful!")
        return True
    except Exception as e:
        print(f" Import error: {e}")
        return False

def test_directories():
    """Check if required directories exist"""
    required_dirs = ['src', 'data', 'models', 'output']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f" Missing directories: {missing_dirs}")
        return False
    else:
        print(" All required directories exist")
        return True

def main():
    """Run all tests"""
    print(" Testing Space Debris Pipeline...")
    print("=" * 40)
    
    # Test 1: Directory structure
    dir_ok = test_directories()
    
    # Test 2: Imports
    import_ok = test_imports()
    
    if dir_ok and import_ok:
        print("\n All tests passed! You can run:")
        print("   python main.py")
    else:
        print("\n Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()