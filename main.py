"""
UIDAI Hackathon - Complete Analysis Pipeline
Main execution script to run the complete analysis pipeline
"""

import os
import sys
from pathlib import Path
import importlib.util

# Add src directory to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def run_complete_analysis():
    """Run the complete analysis pipeline"""
    print("Starting UIDAI Hackathon - Complete Analysis Pipeline")
    print("="*60)
    
    # Create output directories
    Path("outputs/dashboard").mkdir(parents=True, exist_ok=True)
    Path("outputs/ml_models").mkdir(parents=True, exist_ok=True)
    Path("outputs/analysis").mkdir(parents=True, exist_ok=True)
    Path("outputs/policy_analysis").mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Running Phase 1 - Data Exploration...")
    try:
        from src.enhanced_phase1_exploration import main as phase1_main
        phase1_main()
        print("✓ Phase 1 completed")
    except ImportError:
        print("⚠ Phase 1 module not found, skipping...")
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
    
    print("\n[2/4] Running Phase 2 - Machine Learning...")
    try:
        from src.enhanced_phase2_ml import main as phase2_main
        phase2_main()
        print("✓ Phase 2 completed")
    except ImportError:
        print("⚠ Phase 2 module not found, skipping...")
    except Exception as e:
        print(f"✗ Phase 2 failed: {e}")
    
    print("\n[3/4] Running Phase 3 - Dashboard Creation...")
    try:
        from src.enhanced_phase3_dashboard import main as phase3_main
        phase3_main()
        print("✓ Phase 3 completed")
    except ImportError:
        print("⚠ Phase 3 module not found, skipping...")
    except Exception as e:
        print(f"✗ Phase 3 failed: {e}")
    
    print("\n[4/4] Running Policy-Aware Analysis...")
    try:
        from src.policy_aware_analysis import policy_aware_analysis_pipeline
        policy_aware_analysis_pipeline()
        print("✓ Policy-aware analysis completed")
    except ImportError:
        print("⚠ Policy-aware analysis module not found, skipping...")
    except Exception as e:
        print(f"✗ Policy-aware analysis failed: {e}")
    
    print("\n" + "="*60)
    print("Analysis pipeline completed!")
    print("Check outputs/dashboard/ for generated visualizations")
    print("Check outputs/ml_models/ for machine learning results")
    print("Check outputs/policy_analysis/ for policy-aware metrics")

if __name__ == "__main__":
    run_complete_analysis()