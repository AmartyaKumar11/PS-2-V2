# Project Cleanup Script
# This script removes unnecessary files while keeping essential ones for the ML pipeline

import os
import sys

def cleanup_project():
    """Remove unnecessary files from the project folder"""
    
    # Files to remove (unnecessary for ML workflow)
    files_to_remove = [
        'Final.csv',
        'JD_Data_Analyst.txt',
        'JD_Data_Scientist.txt', 
        'JD_Project_Manager.txt',
        'JD_Software_Engineer.txt',
        'react-native-developer--job-description.txt',
        'README.md',
        'GITHUB_SETUP.md',
        'GITHUB_SETUP_INSTRUCTIONS.md',
        'QUICK_COMMANDS.md',
        'setup.py',
        'upload_to_github.bat',
        'skills_it.txt'
    ]
    
    # Essential files to keep (DO NOT REMOVE)
    essential_files = [
        'resume_preprocessing.ipynb',
        'resume_classification_ml.ipynb',
        'parse_resumes_script.py',
        'Web-Developer-job-description.txt',
        'requirements.txt',
        'parsed_resumes.csv',
        'preprocessed_resumes.csv',
        'preprocessed_resumes_lemmatized.csv',
        'resume_matching_results_final.csv',
        'ml_model_comparison_results.csv',
        'final_ml_predictions.csv',
        'feature_importance_analysis.csv',
        'PROJECT_SUMMARY.md'
    ]
    
    removed_files = []
    failed_removals = []
    
    for filename in files_to_remove:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                removed_files.append(filename)
                print(f"✓ Removed: {filename}")
            except Exception as e:
                failed_removals.append((filename, str(e)))
                print(f"✗ Failed to remove {filename}: {e}")
        else:
            print(f"- File not found: {filename}")
    
    print(f"\nCleanup Summary:")
    print(f"Files removed: {len(removed_files)}")
    print(f"Failed removals: {len(failed_removals)}")
    
    print(f"\nEssential files verified:")
    for filename in essential_files:
        if os.path.exists(filename):
            print(f"✓ {filename}")
        else:
            print(f"✗ Missing: {filename}")
    
    return removed_files, failed_removals

if __name__ == "__main__":
    print("Starting project cleanup...")
    removed, failed = cleanup_project()
    print("\nCleanup complete!")
