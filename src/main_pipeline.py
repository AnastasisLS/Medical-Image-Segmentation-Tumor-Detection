"""
Main Pipeline for Medical Image Segmentation Project
Runs all sections in order: Data Pipeline → Segmentation → Detection → Bonus Deep Learning
"""

import os
import sys
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import DataPipeline
from segmentation import SegmentationPipeline
from detection import TumorDetectionPipeline
from bonus_deep_learning import DeepLearningSegmentation


def run_complete_pipeline():
    """
    Run the complete medical image segmentation pipeline.
    """
    print("=" * 80)
    print("MEDICAL IMAGE SEGMENTATION - COMPLETE PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Track overall results
    pipeline_results = {
        'start_time': datetime.now().isoformat(),
        'sections_completed': [],
        'errors': []
    }
    
    try:
        # ============================================================================
        # SECTION 3: DATASET PIPELINE [10 points]
        # ============================================================================
        print("SECTION 3: DATASET PIPELINE")
        print("=" * 50)
        print("Points: 10/100")
        print()
        
        start_time = time.time()
        
        # Initialize data pipeline
        data_pipeline = DataPipeline()
        
        # Task 1: Plot sample subject images
        print("Task 1: Plotting sample subject images...")
        data_pipeline.visualize_sample_subject()
        
        # Task 2: Create train/validation splits
        print("Task 2: Creating train/validation splits...")
        data_pipeline.create_train_val_splits()
        
        # Analyze dataset
        print("Analyzing dataset...")
        data_pipeline.analyze_dataset()
        
        section_time = time.time() - start_time
        print(f"✓ Dataset Pipeline completed in {section_time:.2f} seconds")
        pipeline_results['sections_completed'].append('dataset_pipeline')
        
        print()
        
        # ============================================================================
        # SECTION 4: SEGMENTATION [60 points]
        # ============================================================================
        print("SECTION 4: SEGMENTATION")
        print("=" * 50)
        print("Points: 60/100")
        print()
        
        start_time = time.time()
        
        # Initialize segmentation pipeline
        seg_pipeline = SegmentationPipeline()
        
        # 4.1 Unsupervised Segmentation
        print("4.1 UNSUPERVISED SEGMENTATION")
        print("-" * 30)
        
        # Run unsupervised experiments
        unsupervised_results = seg_pipeline.run_unsupervised_segmentation()
        
        # 4.2 Supervised Segmentation
        print("\n4.2 SUPERVISED SEGMENTATION")
        print("-" * 30)
        
        # Run supervised experiments
        supervised_results = seg_pipeline.run_supervised_segmentation()
        
        # Compare methods
        print("\nCOMPARING SUPERVISED vs UNSUPERVISED")
        print("-" * 30)
        
        # Get sample files for comparison
        sample_files = seg_pipeline.get_files_for_subjects(
            seg_pipeline.val_subjects[:3], max_files_per_subject=1)
        comparison_results = seg_pipeline.compare_methods(sample_files)
        
        section_time = time.time() - start_time
        print(f"✓ Segmentation completed in {section_time:.2f} seconds")
        pipeline_results['sections_completed'].append('segmentation')
        
        print()
        
        # ============================================================================
        # SECTION 5: DETECTING AND COUNTING TUMORS [30 points]
        # ============================================================================
        print("SECTION 5: DETECTING AND COUNTING TUMORS")
        print("=" * 50)
        print("Points: 30/100")
        print()
        
        start_time = time.time()
        
        # Initialize detection pipeline
        detection_pipeline = TumorDetectionPipeline()
        
        # Run complete detection pipeline
        detection_results = detection_pipeline.run_detection_pipeline()
        
        section_time = time.time() - start_time
        print(f"✓ Tumor Detection completed in {section_time:.2f} seconds")
        pipeline_results['sections_completed'].append('detection')
        
        print()
        
        # ============================================================================
        # SECTION 6: BONUS DEEP LEARNING [10 points]
        # ============================================================================
        print("SECTION 6: BONUS DEEP LEARNING")
        print("=" * 50)
        print("Points: 10/100 (Bonus)")
        print()
        
        start_time = time.time()
        
        # Initialize deep learning pipeline
        dl_pipeline = DeepLearningSegmentation()
        
        # Run deep learning pipeline
        dl_results = dl_pipeline.run_deep_learning_pipeline()
        
        if dl_results:
            section_time = time.time() - start_time
            print(f"✓ Deep Learning completed in {section_time:.2f} seconds")
            pipeline_results['sections_completed'].append('deep_learning')
        else:
            print("⚠ Deep Learning failed - continuing with other sections")
            pipeline_results['errors'].append('deep_learning_failed')
        
        print()
        
        # ============================================================================
        # FINAL SUMMARY
        # ============================================================================
        print("=" * 80)
        print("PIPELINE COMPLETION SUMMARY")
        print("=" * 80)
        
        total_points = 0
        completed_sections = len(pipeline_results['sections_completed'])
        
        if 'dataset_pipeline' in pipeline_results['sections_completed']:
            total_points += 10
            print("✓ Dataset Pipeline: 10/10 points")
        
        if 'segmentation' in pipeline_results['sections_completed']:
            total_points += 60
            print("✓ Segmentation: 60/60 points")
        
        if 'detection' in pipeline_results['sections_completed']:
            total_points += 30
            print("✓ Tumor Detection: 30/30 points")
        
        if 'deep_learning' in pipeline_results['sections_completed']:
            total_points += 10
            print("✓ Deep Learning Bonus: 10/10 points")
        
        print(f"\nTotal Points: {total_points}/100")
        print(f"Sections Completed: {completed_sections}/4")
        
        if pipeline_results['errors']:
            print(f"Errors encountered: {len(pipeline_results['errors'])}")
            for error in pipeline_results['errors']:
                print(f"  - {error}")
        
        # Save overall results
        pipeline_results['end_time'] = datetime.now().isoformat()
        pipeline_results['total_points'] = total_points
        pipeline_results['completed_sections'] = completed_sections
        
        # Save to file
        import json
        with open("results/pipeline_summary.json", "w") as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Results saved to: results/pipeline_summary.json")
        print("=" * 80)
        
        return pipeline_results
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        pipeline_results['errors'].append(f"pipeline_failed: {str(e)}")
        pipeline_results['end_time'] = datetime.now().isoformat()
        
        # Save error results
        import json
        with open("results/pipeline_error.json", "w") as f:
            json.dump(pipeline_results, f, indent=2)
        
        return pipeline_results


def main():
    """
    Main function to run the complete pipeline.
    """
    print("Medical Image Segmentation - Complete Pipeline")
    print("This will run all sections of the project:")
    print("1. Dataset Pipeline (10 points)")
    print("2. Segmentation (60 points)")
    print("3. Tumor Detection (30 points)")
    print("4. Bonus Deep Learning (10 points)")
    print()
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠ Warning: Virtual environment may not be activated.")
        print("   Consider running: source venv/bin/activate")
        print()
    
    # Check if data directory exists
    if not os.path.exists("Data"):
        print("❌ Error: Data directory not found!")
        print("   Please ensure the Data/ directory exists with csv/, mask/, and rgb/ subdirectories")
        return
    
    # Check required subdirectories
    required_dirs = ["Data/csv", "Data/mask", "Data/rgb"]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        print(f"❌ Error: Missing required directories: {missing_dirs}")
        return
    
    print("✓ All prerequisites met. Starting pipeline...")
    print()
    
    # Run the complete pipeline
    results = run_complete_pipeline()
    
    if results['errors']:
        print("\n⚠ Some errors occurred during execution.")
        print("Check the results directory for detailed error information.")
    else:
        print("\n🎉 Pipeline completed successfully!")
        print(f"Total points achieved: {results.get('total_points', 0)}/100")


if __name__ == "__main__":
    main() 