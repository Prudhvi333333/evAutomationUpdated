"""
Phase 1: Model Verification Tests
Tests that both TinyLlama (Ollama) and Gemini models are accessible
"""

import os
import sys
import json
from datetime import datetime
import ollama
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configuration
RESULTS_FILE = "test_results.json"
TINYLLAMA_MODEL = "tinyllama"
# Use working Gemini models
GEMINI_MODELS_TO_TEST = [
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-2.5-flash",
]


def test_ollama_connection():
    """Test Ollama is running and list available models."""
    print("\n" + "="*80)
    print("TEST 1: Ollama Connection Check")
    print("="*80)
    
    try:
        # List models
        models = ollama.list()
        available_models = [m['model'] for m in models.get('models', [])]
        
        print(f"✓ Ollama is running")
        print(f"Available models: {available_models}")
        
        return {
            'test': 'ollama_connection',
            'status': 'PASS',
            'available_models': available_models,
            'message': 'Ollama server is accessible'
        }
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {e}")
        return {
            'test': 'ollama_connection',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Ollama server not accessible'
        }


def test_tinyllama_model():
    """Test TinyLlama model is available and can generate."""
    print("\n" + "="*80)
    print("TEST 2: TinyLlama Model Check")
    print("="*80)
    
    try:
        # Try to generate a simple response
        response = ollama.generate(
            model=TINYLLAMA_MODEL,
            prompt="Say 'Hello from TinyLlama' and nothing else.",
            options={'temperature': 0.1, 'num_predict': 50}
        )
        
        answer = response['response'].strip()
        print(f"✓ TinyLlama responded: {answer[:100]}")
        
        return {
            'test': 'tinyllama_model',
            'status': 'PASS',
            'response': answer,
            'message': 'TinyLlama model is working'
        }
    except Exception as e:
        error_msg = str(e)
        print(f"✗ TinyLlama test failed: {error_msg}")
        
        if "not found" in error_msg.lower() or "404" in error_msg:
            print("\n" + "!"*80)
            print("TINYLLAMA NOT FOUND - Run this command to pull it:")
            print("  ollama pull tinyllama")
            print("!"*80)
        
        return {
            'test': 'tinyllama_model',
            'status': 'FAIL',
            'error': error_msg,
            'message': 'TinyLlama model not available'
        }


def test_gemini_api_key():
    """Test Gemini API key is configured."""
    print("\n" + "="*80)
    print("TEST 3: Gemini API Key Check")
    print("="*80)
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("✗ GEMINI_API_KEY not found in environment")
        return {
            'test': 'gemini_api_key',
            'status': 'FAIL',
            'error': 'API key not configured',
            'message': 'Set GEMINI_API_KEY in .env file'
        }
    
    # Mask the key for display
    masked_key = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
    print(f"✓ Gemini API key found: {masked_key}")
    
    return {
        'test': 'gemini_api_key',
        'status': 'PASS',
        'key_present': True,
        'message': 'API key is configured'
    }


def test_gemini_models():
    """Test which Gemini models are available."""
    print("\n" + "="*80)
    print("TEST 4: Gemini Model Availability")
    print("="*80)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("✗ Cannot test Gemini models - API key not configured")
        return {
            'test': 'gemini_models',
            'status': 'SKIP',
            'message': 'API key not configured'
        }
    
    genai.configure(api_key=api_key)
    
    working_models = []
    failed_models = []
    
    for model_name in GEMINI_MODELS_TO_TEST:
        print(f"\n  Testing {model_name}...")
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello' in 1 word.")
            
            if response and response.text:
                print(f"  ✓ {model_name} - WORKING")
                working_models.append({
                    'name': model_name,
                    'response': response.text[:100]
                })
            else:
                print(f"  ✗ {model_name} - Empty response")
                failed_models.append({'name': model_name, 'error': 'Empty response'})
                
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ {model_name} - FAILED: {error_msg[:100]}")
            failed_models.append({'name': model_name, 'error': error_msg})
    
    if working_models:
        print(f"\n✓ {len(working_models)} Gemini model(s) working")
        return {
            'test': 'gemini_models',
            'status': 'PASS',
            'working_models': [m['name'] for m in working_models],
            'failed_models': [m['name'] for m in failed_models],
            'recommended_model': working_models[0]['name'],
            'message': f"Use model: {working_models[0]['name']}"
        }
    else:
        print(f"\n✗ No Gemini models working")
        return {
            'test': 'gemini_models',
            'status': 'FAIL',
            'failed_models': failed_models,
            'message': 'No Gemini models available - check API key or model names'
        }


def run_all_tests():
    """Run all Phase 1 tests."""
    print("\n" + "="*80)
    print("PHASE 1: MODEL VERIFICATION TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'phase': 'phase_1_model_check',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    # Run tests
    results['tests'].append(test_ollama_connection())
    results['tests'].append(test_tinyllama_model())
    results['tests'].append(test_gemini_api_key())
    results['tests'].append(test_gemini_models())
    
    # Calculate summary
    passed = sum(1 for t in results['tests'] if t['status'] == 'PASS')
    failed = sum(1 for t in results['tests'] if t['status'] == 'FAIL')
    skipped = sum(1 for t in results['tests'] if t['status'] == 'SKIP')
    
    results['summary'] = {
        'total': len(results['tests']),
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'can_proceed': passed >= 3  # Need at least 3 tests passing
    }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 SUMMARY")
    print("="*80)
    print(f"Total tests: {len(results['tests'])}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Skipped: {skipped} ⊘")
    print(f"\nCan proceed to Phase 2: {results['summary']['can_proceed']}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    if failed > 0:
        print("\n" + "!"*80)
        print("FIX REQUIRED BEFORE PROCEEDING:")
        for test in results['tests']:
            if test['status'] == 'FAIL':
                print(f"  - {test['test']}: {test.get('message', '')}")
        print("!"*80)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit with appropriate code
    if results['summary']['can_proceed']:
        print("\n✓ Phase 1 complete - Ready for Phase 2")
        sys.exit(0)
    else:
        print("\n✗ Phase 1 incomplete - Fix issues before proceeding")
        sys.exit(1)
