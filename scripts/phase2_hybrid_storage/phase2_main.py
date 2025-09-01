# phase2_hybrid_storage/phase2_main.py
"""
Phase 2: Hybrid Storage System
Main orchestrator for setting up vector + graph hybrid storage

This replaces the monolithic hybrid_storage.py with a modular architecture:
├── 01_vector_db_setup.py      # ChromaDB setup and management
├── 02_embedding_generator.py   # Embedding model with fallbacks  
├── 03_hybrid_storage.py       # Core hybrid storage orchestrator
├── 04_vector_graph_bridge.py  # Advanced vector↔graph bridging
└── phase2_main.py             # This file - main orchestrator
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our modular components - REMOVE the dots!
from phase2_01_vector_db_setup import setup_vector_database
from phase2_02_embedding_generator import setup_embedding_generator  
from phase2_03_hybrid_storage import setup_hybrid_storage
from phase2_04_vector_graph_bridge import setup_vector_graph_bridge

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2Orchestrator:
    """
    Phase 2 Orchestrator
    Manages the complete hybrid storage setup process
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 reset_vector_db: bool = False,
                 embedding_model: Optional[str] = None):
        
        self.config = config or {}
        self.reset_vector_db = reset_vector_db
        self.embedding_model = embedding_model
        
        # Components (will be initialized during setup)
        self.vector_db = None
        self.embedding_generator = None
        self.hybrid_storage = None
        self.bridge = None
        
        # Setup tracking
        self.setup_stats = {
            'start_time': datetime.now(),
            'phases_completed': [],
            'errors': [],
            'final_stats': {}
        }
        
        logger.info("🎬 Phase 2 Orchestrator initialized")
    
    def run_complete_setup(self) -> Dict[str, Any]:
        """Run the complete Phase 2 setup process"""
        
        logger.info("🚀 STARTING PHASE 2: HYBRID STORAGE SYSTEM")
        logger.info("="*60)
        logger.info("Building modular vector + graph hybrid storage")
        logger.info("="*60)
        
        try:
            # Phase 2.1: Vector Database Setup
            self._run_phase_2_1()
            
            # Phase 2.2: Embedding Generator Setup  
            self._run_phase_2_2()
            
            # Phase 2.3: Hybrid Storage Integration
            self._run_phase_2_3()
            
            # Phase 2.4: Vector-Graph Bridge
            self._run_phase_2_4()
            
            # Final validation and stats
            self._finalize_setup()
            
        except Exception as e:
            logger.error(f"❌ Phase 2 setup failed: {e}")
            self.setup_stats['errors'].append(str(e))
            raise
        
        return self.setup_stats
    
    def _run_phase_2_1(self):
        """Phase 2.1: Vector Database Setup"""
        
        logger.info("\n🔸 PHASE 2.1: VECTOR DATABASE SETUP")
        logger.info("-" * 40)
        
        try:
            self.vector_db = setup_vector_database(
                db_path=self.config.get('vector_db_path'),
                reset=self.reset_vector_db
            )
            
            self.setup_stats['phases_completed'].append('2.1_vector_db')
            logger.info("✅ Phase 2.1 Complete")
            
        except Exception as e:
            logger.error(f"❌ Phase 2.1 failed: {e}")
            raise
    
    def _run_phase_2_2(self):
        """Phase 2.2: Embedding Generator Setup"""
        
        logger.info("\n🔸 PHASE 2.2: EMBEDDING GENERATOR SETUP")
        logger.info("-" * 40)
        
        try:
            self.embedding_generator = setup_embedding_generator(
                model_name=self.embedding_model
            )
            
            self.setup_stats['phases_completed'].append('2.2_embedding_generator')
            logger.info("✅ Phase 2.2 Complete")
            
        except Exception as e:
            logger.error(f"❌ Phase 2.2 failed: {e}")
            raise
    
    def _run_phase_2_3(self):
        """Phase 2.3: Hybrid Storage Integration"""
        
        logger.info("\n🔸 PHASE 2.3: HYBRID STORAGE INTEGRATION")
        logger.info("-" * 40)
        
        try:
            # Pass configuration to hybrid storage
            storage_config = {
                'neo4j_uri': self.config.get('neo4j_uri'),
                'neo4j_user': self.config.get('neo4j_user'), 
                'neo4j_password': self.config.get('neo4j_password'),
                'vector_db_path': self.config.get('vector_db_path'),
                'embedding_model': self.embedding_model
            }
            
            self.hybrid_storage = setup_hybrid_storage(**storage_config)
            
            self.setup_stats['phases_completed'].append('2.3_hybrid_storage')
            logger.info("✅ Phase 2.3 Complete")
            
        except Exception as e:
            logger.error(f"❌ Phase 2.3 failed: {e}")
            raise
    
    def _run_phase_2_4(self):
        """Phase 2.4: Vector-Graph Bridge"""
        
        logger.info("\n🔸 PHASE 2.4: VECTOR-GRAPH BRIDGE")
        logger.info("-" * 40)
        
        try:
            self.bridge = setup_vector_graph_bridge(self.hybrid_storage)
            
            self.setup_stats['phases_completed'].append('2.4_vector_graph_bridge')
            logger.info("✅ Phase 2.4 Complete")
            
        except Exception as e:
            logger.error(f"❌ Phase 2.4 failed: {e}")
            raise
    
    def _finalize_setup(self):
        """Finalize setup and gather comprehensive statistics"""
        
        logger.info("\n🔸 FINALIZING PHASE 2 SETUP")
        logger.info("-" * 40)
        
        # Gather comprehensive stats
        if self.hybrid_storage:
            storage_stats = self.hybrid_storage.get_storage_statistics()
            self.setup_stats['final_stats'] = storage_stats
        
        # Calculate setup time
        end_time = datetime.now()
        setup_duration = end_time - self.setup_stats['start_time']
        self.setup_stats['end_time'] = end_time
        self.setup_stats['duration'] = str(setup_duration)
        
        # Log comprehensive results
        logger.info("\n" + "="*60)
        logger.info("🎉 PHASE 2 SETUP COMPLETE!")
        logger.info("="*60)
        
        logger.info(f"⏱️  Setup Duration: {setup_duration}")
        logger.info(f"✅ Phases Completed: {len(self.setup_stats['phases_completed'])}/4")
        
        if self.setup_stats['final_stats']:
            stats = self.setup_stats['final_stats']
            
            # Vector Storage Stats
            vector_stats = stats.get('vector_storage', {})
            logger.info(f"\n📊 VECTOR STORAGE:")
            logger.info(f"   Database Path: {vector_stats.get('db_path', 'Unknown')}")
            logger.info(f"   Collections: {vector_stats.get('total_collections', 0)}")
            logger.info(f"   Total Documents: {vector_stats.get('total_documents', 0):,}")
            
            # Graph Storage Stats  
            graph_stats = stats.get('graph_storage', {})
            logger.info(f"\n🕸️  GRAPH STORAGE:")
            total_nodes = sum(graph_stats.get('nodes', {}).values())
            total_rels = sum(graph_stats.get('relationships', {}).values())
            logger.info(f"   Total Nodes: {total_nodes:,}")
            logger.info(f"   Total Relationships: {total_rels:,}")
            
            # Hybrid Metrics
            hybrid_stats = stats.get('hybrid_metrics', {})
            logger.info(f"\n🌉 HYBRID METRICS:")
            model_info = hybrid_stats.get('embedding_model', {})
            logger.info(f"   Embedding Model: {model_info.get('model_name', 'Unknown')}")
            logger.info(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"   Embedding Dimension: {model_info.get('embedding_dimension', 0)}")
            logger.info(f"   Total Embeddings: {hybrid_stats.get('total_embeddings', 0):,}")
        
        logger.info("\n🎯 SYSTEM READY FOR PHASE 3: QUERY PROCESSING")
        logger.info("="*60)
    
    def run_interactive_demo(self):
        """Run an interactive demo of the hybrid storage system"""
        
        if not self.hybrid_storage or not self.bridge:
            logger.error("❌ System not fully initialized. Run complete setup first.")
            return
        
        logger.info("\n🎮 INTERACTIVE DEMO MODE")
        logger.info("="*40)
        
        demo_queries = [
            "email security and protection features",
            "team collaboration tools",
            "document management and SharePoint", 
            "data loss prevention capabilities",
            "Microsoft Teams integration"
        ]
        
        print("\n🔍 Demo Queries Available:")
        for i, query in enumerate(demo_queries, 1):
            print(f"  {i}. {query}")
        
        print("\n💡 Commands:")
        print("  • Enter a number (1-5) to run demo query")
        print("  • Type your own query")
        print("  • 'stats' to show system statistics")
        print("  • 'clusters' to find semantic clusters")
        print("  • 'recommend <query>' for recommendations")
        print("  • 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n🎯 Enter query or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Demo ended")
                    break
                
                elif user_input.lower() == 'stats':
                    self._show_demo_stats()
                
                elif user_input.lower() == 'clusters':
                    self._show_demo_clusters()
                
                elif user_input.lower().startswith('recommend '):
                    query = user_input[10:]  # Remove 'recommend '
                    self._show_demo_recommendations(query)
                
                elif user_input.isdigit() and 1 <= int(user_input) <= len(demo_queries):
                    query = demo_queries[int(user_input) - 1]
                    self._run_demo_query(query)
                
                elif len(user_input.strip()) > 0:
                    self._run_demo_query(user_input)
                
                else:
                    print("❓ Invalid input. Try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo ended")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _run_demo_query(self, query: str):
        """Run a demo query and display results"""
        
        print(f"\n🔍 Running query: '{query}'")
        print("-" * 50)
        
        try:
            # Use bridge for comprehensive results
            results = self.bridge.semantic_to_structural_query(query, max_results=5)
            
            # Display primary matches
            primary_matches = results.get('primary_matches', [])
            if primary_matches:
                print(f"📊 PRIMARY MATCHES ({len(primary_matches)}):")
                for i, match in enumerate(primary_matches[:3], 1):
                    print(f"  {i}. {match['name']}")
                    print(f"     Type: {match['content_type']}")
                    print(f"     Similarity: {match['vector_similarity']:.3f}")
                    print(f"     Hybrid Score: {match['hybrid_score']:.3f}")
                    print(f"     Graph Connections: {match['graph_connections']}")
                    print()
            
            # Display expanded matches
            expanded_matches = results.get('expanded_matches', [])
            if expanded_matches:
                print(f"🕸️ RELATED ITEMS ({len(expanded_matches)}):")
                for i, match in enumerate(expanded_matches[:5], 1):
                    print(f"  {i}. {match['name']}")
                    print(f"     Connected via: {match['connected_via']}")
                    print(f"     Relationship: {match['relationship_path']}")
                    print(f"     Distance: {match['connection_distance']}")
                    print()
            
            if not primary_matches and not expanded_matches:
                print("❌ No results found. Try a different query.")
                
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    def _show_demo_stats(self):
        """Show system statistics in demo"""
        
        print("\n📊 SYSTEM STATISTICS")
        print("-" * 30)
        
        try:
            stats = self.hybrid_storage.get_storage_statistics()
            
            # Vector stats
            vector_stats = stats.get('vector_storage', {})
            print(f"Vector Database:")
            print(f"  📁 Collections: {vector_stats.get('total_collections', 0)}")
            print(f"  📄 Documents: {vector_stats.get('total_documents', 0):,}")
            
            # Graph stats
            graph_stats = stats.get('graph_storage', {})
            nodes = sum(graph_stats.get('nodes', {}).values())
            rels = sum(graph_stats.get('relationships', {}).values())
            print(f"\nGraph Database:")
            print(f"  🏷️ Nodes: {nodes:,}")
            print(f"  🔗 Relationships: {rels:,}")
            
            # Hybrid stats
            hybrid_stats = stats.get('hybrid_metrics', {})
            model_info = hybrid_stats.get('embedding_model', {})
            print(f"\nHybrid System:")
            print(f"  🧠 Model: {model_info.get('model_name', 'Unknown')}")
            print(f"  📊 Embeddings: {hybrid_stats.get('total_embeddings', 0):,}")
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def _show_demo_clusters(self):
        """Show semantic clusters in demo"""
        
        print("\n🔍 SEMANTIC CLUSTERS")
        print("-" * 25)
        
        try:
            clusters = self.bridge.find_semantic_clusters(
                content_type='features',
                similarity_threshold=0.6,
                min_cluster_size=2
            )
            
            if clusters:
                print(f"Found {len(clusters)} clusters:")
                for i, (cluster_name, items) in enumerate(clusters.items(), 1):
                    print(f"\n  {i}. {cluster_name} ({len(items)} items):")
                    for item in items[:3]:  # Show first 3 items
                        print(f"     • {item['name']}")
            else:
                print("No clusters found with current thresholds.")
                
        except Exception as e:
            print(f"❌ Error finding clusters: {e}")
    
    def _show_demo_recommendations(self, query: str):
        """Show recommendations in demo"""
        
        print(f"\n🎯 RECOMMENDATIONS FOR: '{query}'")
        print("-" * 50)
        
        try:
            recommendations = self.bridge.bridge_recommendation_engine(query)
            
            # Direct matches
            direct = recommendations.get('direct_matches', [])
            if direct:
                print(f"🎯 DIRECT MATCHES:")
                for match in direct[:3]:
                    print(f"  • {match['name']} (score: {match['hybrid_score']:.3f})")
            
            # Suitable plans
            plans = recommendations.get('suitable_plans', [])
            if plans:
                print(f"\n📋 SUITABLE PLANS:")
                for plan in plans[:3]:
                    print(f"  • {plan['name']} (relevance: {plan['relevance_score']:.3f})")
            
            # Related features
            related = recommendations.get('related_features', [])
            if related:
                print(f"\n🔗 RELATED FEATURES:")
                for feature in related[:3]:
                    print(f"  • {feature['name']} (path: {feature['connection_path']})")
            
            # Alternatives
            alternatives = recommendations.get('alternatives', [])
            if alternatives:
                print(f"\n🔄 ALTERNATIVES:")
                for alt in alternatives[:3]:
                    print(f"  • {alt['name']} ({alt['reason']})")
            
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
    
    def export_setup_report(self, output_file: Optional[str] = None) -> str:
        """Export comprehensive setup report"""
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"../reports/phase2_setup_report_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Gather comprehensive report data
        report = {
            'phase': 'Phase 2: Hybrid Storage System',
            'setup_stats': self.setup_stats,
            'system_architecture': {
                'components': [
                    '01_vector_db_setup.py - ChromaDB management',
                    '02_embedding_generator.py - Embedding models with fallbacks',
                    '03_hybrid_storage.py - Core hybrid orchestration', 
                    '04_vector_graph_bridge.py - Vector↔Graph bridging'
                ],
                'data_flow': [
                    'Neo4j Graph → Content Extraction',
                    'Content → Embedding Generation', 
                    'Embeddings → ChromaDB Storage',
                    'Vector + Graph → Hybrid Queries'
                ]
            },
            'final_statistics': self.setup_stats.get('final_stats', {}),
            'recommendations': {
                'next_steps': [
                    'Phase 3: Query Processing Engine',
                    'Phase 4: RAG System Integration',
                    'Phase 5: API and Interface Development'
                ],
                'optimization_tips': [
                    'Monitor embedding model performance',
                    'Tune vector similarity thresholds',
                    'Optimize graph query patterns',
                    'Consider embedding model upgrades'
                ]
            }
        }
        
        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Setup report exported to: {output_path}")
        return str(output_path)
    
    def cleanup(self):
        """Clean up resources"""
        
        logger.info("🧹 Cleaning up Phase 2 resources...")
        
        if self.hybrid_storage:
            self.hybrid_storage.close()
        
        logger.info("✅ Cleanup complete")

def main():
    """Main entry point with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Phase 2: Hybrid Storage System Setup')
    
    parser.add_argument('--reset-vector-db', action='store_true',
                       help='Reset vector database (delete existing collections)')
    parser.add_argument('--embedding-model', type=str, default=None,
                       help='Specific embedding model to use')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demo after setup')
    parser.add_argument('--export-report', action='store_true',
                       help='Export setup report')
    parser.add_argument('--config-file', type=str, default=None,
                       help='JSON config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config_file:
        import json
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Initialize orchestrator
    orchestrator = Phase2Orchestrator(
        config=config,
        reset_vector_db=args.reset_vector_db,
        embedding_model=args.embedding_model
    )
    
    try:
        # Run complete setup
        setup_stats = orchestrator.run_complete_setup()
        
        # Export report if requested
        if args.export_report:
            report_path = orchestrator.export_setup_report()
            print(f"\n📋 Setup report: {report_path}")
        
        # Run demo if requested
        if args.demo:
            orchestrator.run_interactive_demo()
        
        print("\n🎉 Phase 2 setup completed successfully!")
        print("🎯 Ready for Phase 3: Query Processing Engine")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 setup failed: {e}")
        return 1
    
    finally:
        orchestrator.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())