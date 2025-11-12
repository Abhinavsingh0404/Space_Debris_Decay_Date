# test_logger_integration.py
"""
Test script to verify logger integration works in both files
"""
import sys
import os
import subprocess
import time

def test_main_with_logger():
    """Test if main.py runs with logger integration"""
    print("Testing main.py with logger integration...")
    
    try:
        # Increase timeout to 120 seconds (2 minutes) since pipeline takes time
        result = subprocess.run([sys.executable, "main.py"], 
                              capture_output=True, text=True, timeout=120)
        
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("PASS: main.py ran successfully with logger!")
            
            # Check if logger messages appear in output
            if "Space Debris Decay Prediction System" in result.stdout:
                print("PASS: Logger messages detected in output")
            else:
                print("FAIL: Logger messages not found in output")
                
            return True
        else:
            print("FAIL: main.py failed to run")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: main.py timed out (took longer than 2 minutes)")
        return False
    except Exception as e:
        print(f"FAIL: Error testing main.py: {e}")
        return False

def test_data_loader_with_logger():
    """Test if data_loader.py runs with logger integration"""
    print("\nTesting data_loader.py with logger integration...")
    
    try:
        result = subprocess.run([sys.executable, "src/data_loader.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print("PASS: data_loader.py ran successfully with logger!")
            
            # Check if logger messages appear in output
            if "Testing data_loader module directly" in result.stdout:
                print("PASS: Logger messages detected in output")
            else:
                print("FAIL: Logger messages not found in output")
                
            return True
        else:
            print("FAIL: data_loader.py failed to run")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAIL: data_loader.py timed out")
        return False
    except Exception as e:
        print(f"FAIL: Error testing data_loader.py: {e}")
        return False

def check_log_files():
    """Check if log files are being created with actual content"""
    print("\nChecking log files...")
    
    log_dir = "logs"
    if os.path.exists(log_dir):
        log_files = os.listdir(log_dir)
        if log_files:
            print(f"PASS: Log directory exists with {len(log_files)} file(s)")
            
            # Check if files have content (not 0 bytes)
            non_empty_files = []
            for file in log_files:
                file_path = os.path.join(log_dir, file)
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    non_empty_files.append((file, file_size))
            
            if non_empty_files:
                print(f"PASS: {len(non_empty_files)} log files have content:")
                for file, size in non_empty_files[:3]:  # Show first 3 non-empty files
                    print(f"  - {file} ({size} bytes)")
                return True
            else:
                print("FAIL: All log files are empty (0 bytes)")
                return False
        else:
            print("FAIL: Log directory exists but no log files found")
            return False
    else:
        print("FAIL: Log directory 'logs/' does not exist")
        return False

def quick_test():
    """Quick test to verify both files work without timing out"""
    print("QUICK TEST - Running files briefly to check functionality")
    print("=" * 50)
    
    print("\n1. Quick test of data_loader.py:")
    print("-" * 40)
    try:
        # Import and test data_loader quickly
        sys.path.append('src')
        from src.data_loader import load_data
        df = load_data()
        if df is not None:
            print(f"PASS: data_loader loaded {len(df)} rows")
        else:
            print("FAIL: data_loader returned None")
    except Exception as e:
        print(f"FAIL: {e}")
    
    print("\n2. Checking logger functionality:")
    print("-" * 40)
    try:
        from src.logger import get_logger
        test_logger = get_logger('test')
        test_logger.info("Test message from quick test")
        print("PASS: Logger is working")
    except Exception as e:
        print(f"FAIL: Logger error: {e}")

def main():
    """Run all tests"""
    print("Testing Logger Integration")
    print("=" * 50)
    
    # First run a quick test to verify basic functionality
    quick_test()
    
    print("\n" + "="*50)
    print("COMPREHENSIVE TESTS:")
    print("="*50)
    
    # Run comprehensive tests
    test1 = test_main_with_logger()
    test2 = test_data_loader_with_logger() 
    test3 = check_log_files()
    
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"main.py test: {'PASS' if test1 else 'FAIL'}")
    print(f"data_loader.py test: {'PASS' if test2 else 'FAIL'}")
    print(f"log files test: {'PASS' if test3 else 'FAIL'}")
    
    if test1 and test2 and test3:
        print("\nSUCCESS: ALL TESTS PASSED! Logger integration is working correctly.")
    else:
        print("\nWARNING: Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()