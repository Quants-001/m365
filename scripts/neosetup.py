# scripts/neo4j_setup_improved.py
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

if not all([URI, USER, PWD]):
    raise ValueError("Missing Neo4j credentials. Check your .env file.")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

# Updated constraints for M365 data structure
CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Plan) REQUIRE p.id IS UNIQUE", 
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    # Indexes for better performance
    "CREATE INDEX IF NOT EXISTS FOR (f:Feature) ON (f.name)",
    "CREATE INDEX IF NOT EXISTS FOR (p:Plan) ON (p.name)",
    # Full-text search index for RAG system
    "CALL db.index.fulltext.createNodeIndex('featureSearch', ['Feature'], ['name', 'description']) YIELD name RETURN name"
]

def clear_database():
    """Clear all nodes and relationships (be careful!)"""
    with driver.session() as session:
        try:
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleared")
        except Exception as e:
            print(f"Warning during clear: {e}")

def apply_constraints():
    """Apply database constraints and indexes"""
    with driver.session() as session:
        for constraint in CONSTRAINTS:
            try:
                print(f"Applying: {constraint[:50]}...")
                session.run(constraint)
                print("✅ Success")
            except Exception as e:
                print(f"⚠️ Warning: {e}")

def load_m365_data():
    """Load M365 feature-plan data into Neo4j"""
    data_path = Path("../kb/master_data_fixed.csv")
    
    if not data_path.exists():
        print(f"❌ Fixed data file not found: {data_path.resolve()}")
        print("Please run the data fixing script first:")
        print("python scripts/fix_data.py")
        return False
    
    print(f"📁 Loading data from: {data_path.resolve()}")
    
    try:
        df = pd.read_csv(data_path, index_col=0)  # First column is features
        print(f"📊 Data shape: {df.shape[0]} features × {df.shape[1]} plans")
        
        with driver.session() as session:
            # Create Feature nodes
            print("Creating Feature nodes...")
            feature_count = 0
            for feature_name in df.index:
                if pd.notna(feature_name) and str(feature_name).strip():
                    session.run("""
                        CREATE (f:Feature {
                            id: $id,
                            name: $name,
                            description: $name
                        })
                    """, id=f"feat_{hash(feature_name) % 100000}", name=str(feature_name).strip())
                    feature_count += 1
            
            print(f"✅ Created {feature_count} Feature nodes")
            
            # Create Plan nodes
            print("Creating Plan nodes...")
            plan_count = 0
            for plan_name in df.columns:
                if pd.notna(plan_name) and str(plan_name).strip():
                    session.run("""
                        CREATE (p:Plan {
                            id: $id,
                            name: $name,
                            type: $type
                        })
                    """, 
                    id=f"plan_{hash(plan_name) % 100000}", 
                    name=str(plan_name).strip(),
                    type=classify_plan_type(str(plan_name))
                    )
                    plan_count += 1
            
            print(f"✅ Created {plan_count} Plan nodes")
            
            # Create relationships
            print("Creating Feature-Plan relationships...")
            relationship_count = 0
            
            for feature_name in df.index:
                if pd.isna(feature_name) or not str(feature_name).strip():
                    continue
                    
                for plan_name in df.columns:
                    if pd.isna(plan_name) or not str(plan_name).strip():
                        continue
                    
                    availability = df.loc[feature_name, plan_name]
                    
                    # Create relationship if feature is available (marked with ✔ or other indicators)
                    if pd.notna(availability) and str(availability).strip() in ['✔', '✓', 'Yes', 'Available', 'Plan 1', 'Plan 2', 'Basic', 'Standard', 'Premium']:
                        session.run("""
                            MATCH (f:Feature {name: $feature_name})
                            MATCH (p:Plan {name: $plan_name})
                            CREATE (f)-[r:AVAILABLE_IN {
                                status: $status,
                                availability: $availability
                            }]->(p)
                        """, 
                        feature_name=str(feature_name).strip(),
                        plan_name=str(plan_name).strip(),
                        status='available',
                        availability=str(availability).strip()
                        )
                        relationship_count += 1
            
            print(f"✅ Created {relationship_count} relationships")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def classify_plan_type(plan_name):
    """Classify plan into categories for better organization"""
    plan_lower = plan_name.lower()
    
    if any(x in plan_lower for x in ['e1', 'e3', 'e5']):
        return 'Enterprise'
    elif any(x in plan_lower for x in ['f1', 'f3', 'f5']):
        return 'Frontline'
    elif 'business' in plan_lower:
        return 'Business'
    elif any(x in plan_lower for x in ['a1', 'a3', 'a5']):
        return 'Education'
    elif any(x in plan_lower for x in ['basic', 'standard', 'premium']):
        return 'Tier'
    else:
        return 'Other'

def verify_data():
    """Verify the loaded data"""
    with driver.session() as session:
        print("\n📊 Database Summary:")
        
        # Count nodes
        features = session.run("MATCH (f:Feature) RETURN count(f) as count").single()['count']
        plans = session.run("MATCH (p:Plan) RETURN count(p) as count").single()['count'] 
        relationships = session.run("MATCH ()-[r:AVAILABLE_IN]->() RETURN count(r) as count").single()['count']
        
        print(f"  Features: {features}")
        print(f"  Plans: {plans}")
        print(f"  Relationships: {relationships}")
        
        # Sample queries
        print(f"\n🔍 Sample Data:")
        sample_features = session.run("MATCH (f:Feature) RETURN f.name LIMIT 5").data()
        for feat in sample_features:
            print(f"  Feature: {feat['f.name']}")
            
        sample_plans = session.run("MATCH (p:Plan) RETURN p.name, p.type LIMIT 5").data()
        for plan in sample_plans:
            print(f"  Plan: {plan['p.name']} ({plan['p.type']})")

if __name__ == "__main__":
    print("="*60)
    print("NEO4J M365 DATABASE SETUP")
    print("="*60)
    
    # Ask user if they want to clear existing data
    clear = input("Clear existing database? (y/N): ").lower().strip()
    if clear == 'y':
        clear_database()
    
    print("\n1. Applying constraints and indexes...")
    apply_constraints()
    
    print("\n2. Loading M365 data...")
    success = load_m365_data()
    
    if success:
        print("\n3. Verifying data...")
        verify_data()
        
        print("\n" + "="*60)
        print("🎉 NEO4J SETUP COMPLETE!")
        print("Your M365 knowledge graph is ready for RAG!")
        print("="*60)
    else:
        print("\n❌ Setup failed. Please check the errors above.")
    
    # Close driver
    driver.close()