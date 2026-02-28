"""
Phase 2: Data Ingestion Tests
Tests Excel file loading and validation
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Configuration
EXCEL_FILE = "GNEM updated excel (1).xlsx"
RESULTS_FILE = "test_results.json"
EXPECTED_COLUMNS = [
    'Company', 'Category', 'Industry Group', 'Location', 
    'Primary Facility Type', 'EV Supply Chain Role', 'Primary OEMs',
    'Supplier or Affiliation Type', 'Employment', 'Product / Service',
    'EV / Battery Relevant', 'Classification Method'
]


def test_file_exists():
    """Test that Excel file exists."""
    print("\n" + "="*80)
    print("TEST 1: Excel File Exists")
    print("="*80)
    
    if os.path.exists(EXCEL_FILE):
        file_size = os.path.getsize(EXCEL_FILE)
        print(f"✓ File found: {EXCEL_FILE}")
        print(f"  Size: {file_size:,} bytes")
        return {
            'test': 'file_exists',
            'status': 'PASS',
            'file': EXCEL_FILE,
            'size_bytes': file_size,
            'message': 'Excel file is accessible'
        }
    else:
        print(f"✗ File not found: {EXCEL_FILE}")
        return {
            'test': 'file_exists',
            'status': 'FAIL',
            'error': 'File not found',
            'message': f'Ensure {EXCEL_FILE} is in the current directory'
        }


def test_excel_loading():
    """Test that Excel file can be loaded."""
    print("\n" + "="*80)
    print("TEST 2: Excel File Loading")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        print(f"✓ Successfully loaded Excel file")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Column names: {list(df.columns)}")
        
        return {
            'test': 'excel_loading',
            'status': 'PASS',
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'message': 'Excel file loaded successfully'
        }
    except Exception as e:
        print(f"✗ Failed to load Excel: {e}")
        return {
            'test': 'excel_loading',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Failed to load Excel file'
        }


def test_column_structure():
    """Test that required columns exist."""
    print("\n" + "="*80)
    print("TEST 3: Column Structure Validation")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        actual_columns = set(df.columns)
        expected_columns = set(EXPECTED_COLUMNS)
        
        missing = expected_columns - actual_columns
        extra = actual_columns - expected_columns
        
        if not missing:
            print(f"✓ All expected columns present")
            if extra:
                print(f"  Extra columns: {list(extra)}")
            
            return {
                'test': 'column_structure',
                'status': 'PASS',
                'expected_columns': list(expected_columns),
                'actual_columns': list(actual_columns),
                'missing_columns': list(missing),
                'extra_columns': list(extra),
                'message': 'Column structure validated'
            }
        else:
            print(f"✗ Missing columns: {list(missing)}")
            return {
                'test': 'column_structure',
                'status': 'FAIL',
                'missing_columns': list(missing),
                'message': f'Missing required columns: {missing}'
            }
    except Exception as e:
        print(f"✗ Column validation failed: {e}")
        return {
            'test': 'column_structure',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Column validation failed'
        }


def test_data_quality():
    """Test data quality - null values, data types."""
    print("\n" + "="*80)
    print("TEST 4: Data Quality Check")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        
        # Check null values
        null_counts = df.isnull().sum().to_dict()
        null_summary = {k: v for k, v in null_counts.items() if v > 0}
        
        # Check for empty strings
        empty_counts = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                empty_count = (df[col] == '').sum()
                if empty_count > 0:
                    empty_counts[col] = int(empty_count)
        
        print(f"  Total rows: {len(df)}")
        print(f"  Columns with nulls: {len(null_summary)}")
        for col, count in null_summary.items():
            print(f"    - {col}: {count} nulls")
        
        # Sample data check
        sample_rows = df.head(3).to_dict('records')
        
        return {
            'test': 'data_quality',
            'status': 'PASS',
            'total_rows': len(df),
            'null_counts': null_summary,
            'empty_string_counts': empty_counts,
            'sample_data': sample_rows,
            'message': 'Data quality check complete'
        }
    except Exception as e:
        print(f"✗ Data quality check failed: {e}")
        return {
            'test': 'data_quality',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Data quality check failed'
        }


def test_key_fields():
    """Test that key fields have valid data."""
    print("\n" + "="*80)
    print("TEST 5: Key Fields Validation")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        
        results = {}
        
        # Check Company field
        if 'Company' in df.columns:
            unique_companies = df['Company'].nunique()
            null_companies = df['Company'].isnull().sum()
            print(f"  Company field: {unique_companies} unique, {null_companies} nulls")
            results['company'] = {'unique': int(unique_companies), 'nulls': int(null_companies)}
        
        # Check EV Supply Chain Role
        if 'EV Supply Chain Role' in df.columns:
            roles = df['EV Supply Chain Role'].value_counts().to_dict()
            print(f"  EV Roles: {list(roles.keys())}")
            results['ev_roles'] = roles
        
        # Check Employment (should be numeric)
        if 'Employment' in df.columns:
            numeric_employment = pd.to_numeric(df['Employment'], errors='coerce').notna().sum()
            print(f"  Employment: {numeric_employment}/{len(df)} numeric values")
            results['employment_numeric'] = int(numeric_employment)
        
        return {
            'test': 'key_fields',
            'status': 'PASS',
            'field_stats': results,
            'message': 'Key fields validated'
        }
    except Exception as e:
        print(f"✗ Key fields validation failed: {e}")
        return {
            'test': 'key_fields',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Key fields validation failed'
        }


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n" + "="*80)
    print("PHASE 2: DATA INGESTION TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing file: {EXCEL_FILE}")
    
    results = {
        'phase': 'phase_2_ingestion',
        'timestamp': datetime.now().isoformat(),
        'file': EXCEL_FILE,
        'tests': []
    }
    
    # Run tests
    results['tests'].append(test_file_exists())
    results['tests'].append(test_excel_loading())
    results['tests'].append(test_column_structure())
    results['tests'].append(test_data_quality())
    results['tests'].append(test_key_fields())
    
    # Calculate summary
    passed = sum(1 for t in results['tests'] if t['status'] == 'PASS')
    failed = sum(1 for t in results['tests'] if t['status'] == 'FAIL')
    
    results['summary'] = {
        'total': len(results['tests']),
        'passed': passed,
        'failed': failed,
        'can_proceed': failed == 0
    }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 2 SUMMARY")
    print("="*80)
    print(f"Total tests: {len(results['tests'])}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"\nCan proceed to Phase 3: {results['summary']['can_proceed']}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    return results


if __name__ == "__main__":
    # Change to project root to find Excel file
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    results = run_all_tests()
    
    if results['summary']['can_proceed']:
        print("\n✓ Phase 2 complete - Ready for Phase 3")
        sys.exit(0)
    else:
        print("\n✗ Phase 2 incomplete - Fix issues before proceeding")
        sys.exit(1)
