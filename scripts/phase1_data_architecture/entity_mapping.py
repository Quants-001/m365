# scripts/entity_linking.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
import logging

# Neo4j for graph operations
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityLinker:
    """
    Advanced entity linking system for M365 features and plans.
    Handles:
    1. Entity resolution (finding canonical names)
    2. Duplicate detection and merging
    3. Semantic mapping between different data sources
    4. Ambiguity resolution
    """
    
    def __init__(self):
        load_dotenv()
        
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Load existing mappings if available
        self.entity_mappings = self._load_existing_mappings()
        
        # Similarity thresholds
        self.FEATURE_SIMILARITY_THRESHOLD = 0.85
        self.PLAN_SIMILARITY_THRESHOLD = 0.90
        
        logger.info("Entity Linker initialized")
    
    def _load_existing_mappings(self) -> Dict:
        """Load existing entity mappings from file"""
        mapping_path = Path("../data/entity_mappings.json")
        
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                return json.load(f)
        
        return {
            'feature_aliases': {},
            'plan_aliases': {},
            'canonical_features': {},
            'canonical_plans': {},
            'ambiguous_entities': []
        }
    
    def _save_mappings(self):
        """Save entity mappings to file"""
        mapping_path = Path("../data/entity_mappings.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_path, 'w') as f:
            json.dump(self.entity_mappings, f, indent=2)
        
        logger.info(f"Entity mappings saved to {mapping_path}")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if pd.isna(text):
            return ""
        
        # Convert to string and normalize
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters for comparison
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Use SequenceMatcher for fuzzy matching
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        
        # Boost similarity for exact substring matches
        if norm1 in norm2 or norm2 in norm1:
            similarity = max(similarity, 0.8)
        
        # Boost similarity for keyword matches
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if words1 & words2:  # Common words
            word_similarity = len(words1 & words2) / max(len(words1), len(words2))
            similarity = max(similarity, word_similarity * 0.7)
        
        return similarity
    
    def extract_entities_from_graph(self) -> Dict[str, List[Dict]]:
        """Extract existing entities from Neo4j graph"""
        entities = {
            'features': [],
            'plans': [],
            'categories': []
        }
        
        with self.neo4j_driver.session() as session:
            # Extract features
            feature_query = """
            MATCH (f:Feature)
            OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
            RETURN f.id as id, f.name as name, f.description as description,
                   c.name as category
            """
            
            features = session.run(feature_query).data()
            for feature in features:
                entities['features'].append({
                    'id': feature['id'],
                    'name': feature['name'],
                    'description': feature.get('description', ''),
                    'category': feature.get('category', ''),
                    'normalized_name': self.normalize_text(feature['name'])
                })
            
            # Extract plans
            plan_query = """
            MATCH (p:Plan)
            RETURN p.id as id, p.name as name, p.type as type
            """
            
            plans = session.run(plan_query).data()
            for plan in plans:
                entities['plans'].append({
                    'id': plan['id'],
                    'name': plan['name'],
                    'type': plan.get('type', ''),
                    'normalized_name': self.normalize_text(plan['name'])
                })
            
            # Extract categories
            category_query = """
            MATCH (c:Category)
            RETURN c.id as id, c.name as name, c.type as type
            """
            
            categories = session.run(category_query).data()
            for category in categories:
                entities['categories'].append({
                    'id': category['id'],
                    'name': category['name'],
                    'type': category.get('type', ''),
                    'normalized_name': self.normalize_text(category['name'])
                })
        
        logger.info(f"Extracted {len(entities['features'])} features, "
                   f"{len(entities['plans'])} plans, "
                   f"{len(entities['categories'])} categories from graph")
        
        return entities
    
    def find_similar_entities(self, entities: List[Dict], threshold: float = 0.85) -> List[List[Dict]]:
        """Find groups of similar entities"""
        similar_groups = []
        processed = set()
        
        for i, entity1 in enumerate(entities):
            if i in processed:
                continue
            
            similar_group = [entity1]
            processed.add(i)
            
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_similarity(
                    entity1['normalized_name'], 
                    entity2['normalized_name']
                )
                
                if similarity >= threshold:
                    similar_group.append(entity2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
        
        return similar_groups
    
    def resolve_feature_entities(self) -> Dict:
        """Resolve and deduplicate feature entities"""
        logger.info("Resolving feature entities...")
        
        entities = self.extract_entities_from_graph()
        features = entities['features']
        
        # Find similar features
        similar_feature_groups = self.find_similar_entities(
            features, 
            self.FEATURE_SIMILARITY_THRESHOLD
        )
        
        resolution_results = {
            'canonical_features': {},
            'merged_features': [],
            'ambiguous_features': []
        }
        
        for group in similar_feature_groups:
            # Choose canonical name (longest, most descriptive)
            canonical = max(group, key=lambda x: len(x['name']))
            canonical_id = canonical['id']
            canonical_name = canonical['name']
            
            # Map all variants to canonical
            for feature in group:
                resolution_results['canonical_features'][feature['name']] = {
                    'canonical_id': canonical_id,
                    'canonical_name': canonical_name,
                    'original_id': feature['id'],
                    'similarity_score': self.calculate_similarity(
                        feature['name'], canonical_name
                    )
                }
            
            resolution_results['merged_features'].append({
                'canonical_id': canonical_id,
                'canonical_name': canonical_name,
                'variants': [f['name'] for f in group],
                'merge_count': len(group)
            })
            
            logger.info(f"Merged {len(group)} feature variants: {canonical_name}")
        
        return resolution_results
    
    def resolve_plan_entities(self) -> Dict:
        """Resolve and deduplicate plan entities"""
        logger.info("Resolving plan entities...")
        
        entities = self.extract_entities_from_graph()
        plans = entities['plans']
        
        # Find similar plans
        similar_plan_groups = self.find_similar_entities(
            plans, 
            self.PLAN_SIMILARITY_THRESHOLD
        )
        
        resolution_results = {
            'canonical_plans': {},
            'merged_plans': [],
            'plan_hierarchy': []
        }
        
        for group in similar_plan_groups:
            # For plans, prefer shorter, more official names
            canonical = min(group, key=lambda x: len(x['name']))
            canonical_id = canonical['id']
            canonical_name = canonical['name']
            
            # Map all variants to canonical
            for plan in group:
                resolution_results['canonical_plans'][plan['name']] = {
                    'canonical_id': canonical_id,
                    'canonical_name': canonical_name,
                    'original_id': plan['id'],
                    'plan_type': plan['type'],
                    'similarity_score': self.calculate_similarity(
                        plan['name'], canonical_name
                    )
                }
            
            resolution_results['merged_plans'].append({
                'canonical_id': canonical_id,
                'canonical_name': canonical_name,
                'variants': [p['name'] for p in group],
                'merge_count': len(group)
            })
            
            logger.info(f"Merged {len(group)} plan variants: {canonical_name}")
        
        return resolution_results
    
    def create_entity_aliases(self, feature_results: Dict, plan_results: Dict):
        """Create comprehensive entity alias mappings"""
        logger.info("Creating entity aliases...")
        
        # Feature aliases
        for original_name, mapping in feature_results['canonical_features'].items():
            canonical_name = mapping['canonical_name']
            
            if canonical_name not in self.entity_mappings['feature_aliases']:
                self.entity_mappings['feature_aliases'][canonical_name] = []
            
            if original_name not in self.entity_mappings['feature_aliases'][canonical_name]:
                self.entity_mappings['feature_aliases'][canonical_name].append(original_name)
        
        # Plan aliases
        for original_name, mapping in plan_results['canonical_plans'].items():
            canonical_name = mapping['canonical_name']
            
            if canonical_name not in self.entity_mappings['plan_aliases']:
                self.entity_mappings['plan_aliases'][canonical_name] = []
            
            if original_name not in self.entity_mappings['plan_aliases'][canonical_name]:
                self.entity_mappings['plan_aliases'][canonical_name].append(original_name)
        
        # Update canonical mappings
        self.entity_mappings['canonical_features'].update(feature_results['canonical_features'])
        self.entity_mappings['canonical_plans'].update(plan_results['canonical_plans'])
    
    def apply_entity_resolution_to_graph(self):
        """Apply entity resolution results to the Neo4j graph using a robust method."""
        logger.info("Applying entity resolution to graph...")
        
        with self.neo4j_driver.session() as session:
            # Note: This process works best with the APOC library installed on your Neo4j database.
            # It provides the apoc.refactor.mergeNodes procedure for safe node merging.

            # 1. Merge duplicate features
            for canonical_name, aliases in self.entity_mappings['feature_aliases'].items():
                if len(aliases) <= 1:
                    continue

                # Get all alias names excluding the canonical one
                alias_names = [name for name in aliases if name != canonical_name]
                if not alias_names:
                    continue

                try:
                    # This query finds the canonical node and all alias nodes, then merges them.
                    # apoc.refactor.mergeNodes handles moving all relationships safely.
                    session.run("""
                        MATCH (canonical:Feature {name: $canonical_name})
                        WITH canonical
                        MATCH (alias:Feature)
                        WHERE alias.name IN $alias_names
                        WITH canonical, collect(alias) AS aliasNodes
                        CALL apoc.refactor.mergeNodes([canonical] + aliasNodes, {
                            properties: 'discard', 
                            mergeRels: true
                        })
                        YIELD node
                        RETURN count(node)
                    """, canonical_name=canonical_name, alias_names=alias_names)
                    logger.info(f"Merged {len(alias_names)} feature aliases into: {canonical_name}")
                except Exception as e:
                    logger.error(f"Could not merge features for '{canonical_name}': {e}")
                    logger.warning("Consider installing the APOC plugin on your Neo4j database for a more robust merge.")


            # 2. Merge duplicate plans
            for canonical_name, aliases in self.entity_mappings['plan_aliases'].items():
                if len(aliases) <= 1:
                    continue
                
                alias_names = [name for name in aliases if name != canonical_name]
                if not alias_names:
                    continue
                
                try:
                    session.run("""
                        MATCH (canonical:Plan {name: $canonical_name})
                        WITH canonical
                        MATCH (alias:Plan)
                        WHERE alias.name IN $alias_names
                        WITH canonical, collect(alias) AS aliasNodes
                        CALL apoc.refactor.mergeNodes([canonical] + aliasNodes, {
                            properties: 'discard', 
                            mergeRels: true
                        })
                        YIELD node
                        RETURN count(node)
                    """, canonical_name=canonical_name, alias_names=alias_names)
                    logger.info(f"Merged {len(alias_names)} plan aliases into: {canonical_name}")
                except Exception as e:
                    logger.error(f"Could not merge plans for '{canonical_name}': {e}")
                    logger.warning("Consider installing the APOC plugin on your Neo4j database for a more robust merge.")
    
    def create_semantic_links(self):
        """Create semantic links between entities"""
        logger.info("Creating semantic links...")
        
        with self.neo4j_driver.session() as session:
            # Link features with common keywords
            semantic_rules = [
                # Security-related features
                {
                    'keywords': ['security', 'defender', 'threat', 'protection', 'compliance'],
                    'concept': 'Security',
                    'relationship': 'RELATED_TO_SECURITY'
                },
                # Collaboration features
                {
                    'keywords': ['teams', 'sharepoint', 'collaboration', 'meeting', 'chat'],
                    'concept': 'Collaboration',
                    'relationship': 'ENABLES_COLLABORATION'
                },
                # Productivity features
                {
                    'keywords': ['office', 'word', 'excel', 'powerpoint', 'onenote', 'copilot'],
                    'concept': 'Productivity',
                    'relationship': 'ENHANCES_PRODUCTIVITY'
                },
                # Analytics features
                {
                    'keywords': ['analytics', 'insights', 'viva', 'reporting', 'dashboard'],
                    'concept': 'Analytics',
                    'relationship': 'PROVIDES_ANALYTICS'
                }
            ]
            
            for rule in semantic_rules:
                # Create concept node if it doesn't exist
                session.run("""
                    MERGE (c:Concept {name: $concept})
                    SET c.type = 'Semantic'
                """, concept=rule['concept'])
                
                # Link features containing keywords
                for keyword in rule['keywords']:
                    result = session.run(f"""
                        MATCH (f:Feature)
                        MATCH (c:Concept {{name: $concept}})
                        WHERE toLower(f.name) CONTAINS toLower($keyword)
                        MERGE (f)-[r:{rule['relationship']}]->(c)
                        RETURN count(r) as count
                    """, keyword=keyword, concept=rule['concept'])
                    
                    count = result.single()['count']
                    if count > 0:
                        logger.info(f"Created {count} {rule['relationship']} links for '{keyword}'")
    
    def generate_entity_report(self) -> Dict:
        """Generate comprehensive entity linking report"""
        logger.info("Generating entity linking report...")
        
        report = {
            'summary': {
                'total_features': len(self.entity_mappings['canonical_features']),
                'total_plans': len(self.entity_mappings['canonical_plans']),
                'feature_aliases': len(self.entity_mappings['feature_aliases']),
                'plan_aliases': len(self.entity_mappings['plan_aliases']),
                'ambiguous_entities': len(self.entity_mappings['ambiguous_entities'])
            },
            'top_merged_features': [],
            'top_merged_plans': [],
            'quality_metrics': {}
        }
        
        # Find most merged entities
        feature_merge_counts = {
            canonical: len(aliases) for canonical, aliases 
            in self.entity_mappings['feature_aliases'].items()
        }
        
        report['top_merged_features'] = sorted(
            feature_merge_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        plan_merge_counts = {
            canonical: len(aliases) for canonical, aliases 
            in self.entity_mappings['plan_aliases'].items()
        }
        
        report['top_merged_plans'] = sorted(
            plan_merge_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Quality metrics
        total_feature_merges = sum(feature_merge_counts.values())
        total_plan_merges = sum(plan_merge_counts.values())
        
        report['quality_metrics'] = {
            'average_feature_merges': total_feature_merges / max(len(feature_merge_counts), 1),
            'average_plan_merges': total_plan_merges / max(len(plan_merge_counts), 1),
            'deduplication_ratio': {
                'features': total_feature_merges / max(len(self.entity_mappings['canonical_features']), 1),
                'plans': total_plan_merges / max(len(self.entity_mappings['canonical_plans']), 1)
            }
        }
        
        return report
    
    def run_complete_entity_linking(self) -> Dict:
        """Run the complete entity linking process"""
        logger.info("Starting complete entity linking process...")
        
        # 1. Resolve feature entities
        feature_results = self.resolve_feature_entities()
        
        # 2. Resolve plan entities
        plan_results = self.resolve_plan_entities()
        
        # 3. Create alias mappings
        self.create_entity_aliases(feature_results, plan_results)
        
        # 4. Apply resolution to graph
        self.apply_entity_resolution_to_graph()
        
        # 5. Create semantic links
        self.create_semantic_links()
        
        # 6. Save mappings
        self._save_mappings()
        
        # 7. Generate report
        report = self.generate_entity_report()
        
        logger.info("Entity linking process completed")
        return report

def main():
    """Main function to run entity linking"""
    logger.info("="*60)
    logger.info("ENTITY LINKING AND RESOLUTION")
    logger.info("="*60)
    
    try:
        linker = EntityLinker()
        report = linker.run_complete_entity_linking()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ENTITY LINKING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Features: {report['summary']['total_features']}")
        logger.info(f"Total Plans: {report['summary']['total_plans']}")
        logger.info(f"Feature Aliases: {report['summary']['feature_aliases']}")
        logger.info(f"Plan Aliases: {report['summary']['plan_aliases']}")
        
        if report['top_merged_features']:
            logger.info(f"\nTop Merged Features:")
            for canonical, count in report['top_merged_features'][:5]:
                logger.info(f"  {canonical}: {count} variants")
        
        if report['top_merged_plans']:
            logger.info(f"\nTop Merged Plans:")
            for canonical, count in report['top_merged_plans'][:5]:
                logger.info(f"  {canonical}: {count} variants")
        
        logger.info("\n" + "="*60)
        logger.info("ENTITY LINKING COMPLETE!")
        logger.info("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"Error in entity linking: {e}")
        raise

if __name__ == "__main__":
    report = main()
    print(f"\nEntity linking completed with {report['summary']['total_features']} features and {report['summary']['total_plans']} plans")