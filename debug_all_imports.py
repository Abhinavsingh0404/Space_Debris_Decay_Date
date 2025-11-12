# debug_all_imports.py
import sys
import os

print("=== DEBUGGING ALL IMPORTS ===")

# Add current directory to path
sys.path.insert(0, os.getcwd())

modules_to_check = [
    ('data_loader', ['load_data', 'create_target_variable']),
    ('preprocessing', ['handle_missing_values', 'engineer_features', 'prepare_features', 'scale_features']),
    ('model_training', ['compare_models', 'train_final_model']),
    ('pipeline', ['run_full_pipeline'])
]

all_ok = True

for module_name, functions in modules_to_check:
    print(f"\n--- Checking {module_name} ---")
    try:
        module = __import__(f'src.{module_name}', fromlist=functions)
        print(f"✅ {module_name} imported successfully")
        
        for func in functions:
            if hasattr(module, func):
                print(f"  ✅ {func} found")
            else:
                print(f"  ❌ {func} NOT found")
                all_ok = False
                
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        all_ok = False
    except Exception as e:
        print(f"❌ Error with {module_name}: {e}")
        all_ok = False

print(f"\n=== OVERALL RESULT: {'✅ ALL IMPORTS OK' if all_ok else '❌ SOME IMPORTS FAILED'} ===")
