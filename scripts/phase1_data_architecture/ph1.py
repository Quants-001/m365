import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    """Runs a Python script and checks for errors."""
    print("\n" + "="*80)
    print(f"🚀 EXECUTING: {script_path.name}")
    print("="*80)
    
    absolute_path = script_path.resolve()
    python_executable = sys.executable
    
    # Run the script from its own directory to resolve relative paths like ../kb/
    working_dir = script_path.parent
    
    result = subprocess.run([python_executable, str(absolute_path)], capture_output=True, text=True, cwd=working_dir)
    
    if result.returncode != 0:
        print(f"❌ ERROR IN SCRIPT: {script_path.name}")
        print("------- STDOUT -------")
        print(result.stdout)
        print("------- STDERR -------")
        print(result.stderr)
        print("="*80)
        return False
    else:
        print(f"✅ SUCCESS: {script_path.name} finished.")
        print("------- SCRIPT OUTPUT (Last 20 lines) -------")
        output_lines = result.stdout.strip().split('\n')
        for line in output_lines[-20:]:
            print(line)
        print("="*80)
        return True

def main():
    """Runs the complete Phase 1 data architecture pipeline in the correct order."""
    
    base_path = Path("D:\m365\scripts\phase1_data_architecture")
    
    # This is the correct, logical sequence for building your graph
    phase1_scripts = [
        # Step 0 (Optional, if you need to re-clean the CSV)
        # base_path / "fix_data.py", 
        
        # Step 1: Clear the database and do the initial load of raw data
        base_path / "neosetup.py",
        
        # Step 2: Enrich the graph with categories, concepts, and relationships
        base_path / "knowledge_graph.py",
        
        # Step 3: Clean the graph by merging duplicate entities
        base_path / "entity_linking.py",
        
        # Step 4: Index the final, clean, and enriched graph for performance
        base_path / "graph_indexing.py"
    ]
    
    for script in phase1_scripts:
        if not script.exists():
            print(f"🚨 FATAL ERROR: Script not found at {script}. Aborting pipeline.")
            return
            
        if not run_script(script):
            print(f"Pipeline stopped due to error in {script.name}.")
            return
            
    print("\n🎉🎉🎉 COMPLETE PHASE 1 PIPELINE FINISHED SUCCESSFULLY! 🎉🎉🎉")
    print("Your knowledge graph is now fully built, enriched, cleaned, and indexed.")

if __name__ == "__main__":
    main()

