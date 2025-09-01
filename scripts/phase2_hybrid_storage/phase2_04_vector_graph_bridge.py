# phase2_hybrid_storage/04_vector_graph_bridge.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Import our modular components
from phase2_03_hybrid_storage import HybridStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorGraphBridge:
    """
    Vector-Graph Bridge System
    Creates intelligent connections between vector similarity and graph traversal
    """
    
    def __init__(self, hybrid_storage: HybridStorage):
        """Initialize the bridge with hybrid storage system"""
        
        self.hybrid_storage = hybrid_storage
        self.mapping_file = Path("../data/vector_graph_mapping.json")
        self.bridge_cache = {}
        
        # Create or load vector-graph mappings
        self.vector_graph_mapping = self._create_mapping()
        
        logger.info("🌉 Vector-Graph Bridge initialized")
    
    def _create_mapping(self) -> Dict[str, Dict]:
        """Create bidirectional mapping between vector IDs and graph node IDs"""
        
        logger.info("🔗 Creating vector-graph mappings...")
        
        mapping = {
            'vector_to_graph': {
                'features': {},
                'plans': {},
                'relationships': {}
            },
            'graph_to_vector': {
                'features': {},
                'plans': {}
            },
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_mappings': 0
            }
        }
        
        with self.hybrid_storage.neo4j_driver.session() as session:
            # Map Features
            features = session.run("MATCH (f:Feature) RETURN f.id as id, f.name as name").data()
            for feature in features:
                neo4j_id = feature['id'] or feature['name']  # Fallback to name if no ID
                vector_id = self._find_vector_id_by_name(feature['name'], 'features')
                
                if vector_id:
                    mapping['vector_to_graph']['features'][vector_id] = neo4j_id
                    mapping['graph_to_vector']['features'][neo4j_id] = vector_id
            
            # Map Plans
            plans = session.run("MATCH (p:Plan) RETURN p.id as id, p.name as name").data()
            for plan in plans:
                neo4j_id = plan['id'] or plan['name']  # Fallback to name if no ID
                vector_id = self._find_vector_id_by_name(plan['name'], 'plans')
                
                if vector_id:
                    mapping['vector_to_graph']['plans'][vector_id] = neo4j_id
                    mapping['graph_to_vector']['plans'][neo4j_id] = vector_id
        
        # Calculate total mappings
        total_mappings = (len(mapping['vector_to_graph']['features']) + 
                         len(mapping['vector_to_graph']['plans']))
        mapping['metadata']['total_mappings'] = total_mappings
        
        # Save mapping to file
        self.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"✅ Created {total_mappings} vector-graph mappings")
        logger.info(f"📁 Saved to: {self.mapping_file}")
        
        return mapping
    
    def _find_vector_id_by_name(self, name: str, content_type: str) -> Optional[str]:
        """Find vector ID by searching for matching name in collection"""
        
        collection_name = f'm365_{content_type}'
        if collection_name not in self.hybrid_storage.collections:
            return None
        
        collection = self.hybrid_storage.collections[collection_name]
        
        try:
            # Search for exact name match in metadata
            results = collection.get(
                where={"name": name},
                include=['ids']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return results['ids'][0]
                
        except Exception as e:
            logger.debug(f"Could not find vector ID for {name}: {e}")
        
        return None
    
    def semantic_to_structural_query(self, 
                                   query: str, 
                                   expand_depth: int = 2,
                                   max_results: int = 10) -> Dict[str, Any]:
        """
        Convert semantic vector query to structural graph traversal
        1. Find semantically similar items via vector search
        2. Expand to connected graph nodes
        3. Return enriched results
        """
        
        logger.info(f"🔍 Semantic→Structural query: '{query}'")
        
        # Step 1: Vector similarity search
        vector_results = self.hybrid_storage.query_hybrid(
            query, 
            n_results=max_results//2,
            similarity_threshold=0.1
        )
        
        # Step 2: Convert vector hits to graph nodes
        graph_expansions = {}
        
        for content_type, items in vector_results.items():
            if not items:
                continue
            
            graph_expansions[content_type] = []
            
            for item in items:
                item_name = item['metadata'].get('name', '')
                if not item_name:
                    continue
                
                # Get graph context for this item
                if content_type == 'features':
                    graph_context = self.hybrid_storage.get_graph_context(item_name, 'Feature')
                elif content_type == 'plans':
                    graph_context = self.hybrid_storage.get_graph_context(item_name, 'Plan')
                else:
                    graph_context = {}
                
                # Combine vector similarity with graph structure
                enriched_result = {
                    'vector_match': item,
                    'graph_context': graph_context,
                    'expansion_paths': self._get_expansion_paths(item_name, content_type, expand_depth)
                }
                
                graph_expansions[content_type].append(enriched_result)
        
        # Step 3: Cross-reference and rank results
        final_results = self._rank_hybrid_results(graph_expansions, query)
        
        return final_results
    
    def _get_expansion_paths(self, 
                           item_name: str, 
                           content_type: str, 
                           depth: int = 2) -> List[Dict]:
        """Get expansion paths from a given item in the graph"""
        
        if depth <= 0:
            return []
        
        expansion_paths = []
        
        with self.hybrid_storage.neo4j_driver.session() as session:
            if content_type == 'features':
                # Expand from feature to plans, categories, and related features
                query = """
                MATCH (start:Feature {name: $name})
                MATCH paths = (start)-[*1..2]-(connected)
                WHERE NOT start = connected
                RETURN DISTINCT connected, 
                       type(connected) as node_type,
                       length(paths) as distance,
                       [r in relationships(paths) | type(r)] as relationship_types
                LIMIT 10
                """
            elif content_type == 'plans':
                # Expand from plan to features and related plans
                query = """
                MATCH (start:Plan {name: $name})
                MATCH paths = (start)-[*1..2]-(connected)
                WHERE NOT start = connected
                RETURN DISTINCT connected,
                       labels(connected)[0] as node_type,
                       length(paths) as distance,
                       [r in relationships(paths) | type(r)] as relationship_types
                LIMIT 10
                """
            else:
                return []
            
            try:
                results = session.run(query, name=item_name).data()
                
                for result in results:
                    connected_node = dict(result['connected'].items())
                    
                    expansion_paths.append({
                        'connected_item': connected_node,
                        'node_type': result['node_type'],
                        'distance': result['distance'],
                        'relationship_path': result['relationship_types']
                    })
                    
            except Exception as e:
                logger.debug(f"Error getting expansion paths for {item_name}: {e}")
        
        return expansion_paths
    
    def _rank_hybrid_results(self, 
                           graph_expansions: Dict[str, List], 
                           original_query: str) -> Dict[str, Any]:
        """Rank results combining vector similarity and graph structure"""
        
        ranked_results = {
            'primary_matches': [],
            'expanded_matches': [],
            'cross_references': [],
            'query_summary': {
                'original_query': original_query,
                'total_primary': 0,
                'total_expanded': 0,
                'processing_time': datetime.now().isoformat()
            }
        }
        
        # Process primary matches (direct vector hits)
        for content_type, items in graph_expansions.items():
            for item in items:
                vector_match = item['vector_match']
                graph_context = item.get('graph_context', {})
                
                # Calculate hybrid score
                hybrid_score = self._calculate_hybrid_score(
                    vector_match['similarity'],
                    graph_context,
                    item.get('expansion_paths', [])
                )
                
                primary_match = {
                    'name': vector_match['metadata'].get('name', 'Unknown'),
                    'content_type': content_type,
                    'vector_similarity': vector_match['similarity'],
                    'hybrid_score': hybrid_score,
                    'text': vector_match['text'][:200] + "...",
                    'graph_connections': len(item.get('expansion_paths', [])),
                    'metadata': vector_match['metadata']
                }
                
                ranked_results['primary_matches'].append(primary_match)
        
        # Process expanded matches (from graph traversal)
        for content_type, items in graph_expansions.items():
            for item in items:
                for expansion in item.get('expansion_paths', []):
                    connected_item = expansion['connected_item']
                    
                    expanded_match = {
                        'name': connected_item.get('name', 'Unknown'),
                        'content_type': expansion['node_type'].lower() + 's',  # Convert to plural
                        'connection_distance': expansion['distance'],
                        'relationship_path': ' → '.join(expansion['relationship_path']),
                        'connected_via': item['vector_match']['metadata'].get('name', 'Unknown'),
                        'indirect_relevance': 1.0 / (expansion['distance'] + 1)  # Closer = more relevant
                    }
                    
                    ranked_results['expanded_matches'].append(expanded_match)
        
        # Sort results by hybrid score
        ranked_results['primary_matches'] = sorted(
            ranked_results['primary_matches'], 
            key=lambda x: x['hybrid_score'], 
            reverse=True
        )
        
        ranked_results['expanded_matches'] = sorted(
            ranked_results['expanded_matches'],
            key=lambda x: x['indirect_relevance'],
            reverse=True
        )[:20]  # Limit expanded results
        
        # Update summary
        ranked_results['query_summary']['total_primary'] = len(ranked_results['primary_matches'])
        ranked_results['query_summary']['total_expanded'] = len(ranked_results['expanded_matches'])
        
        return ranked_results
    
    def _calculate_hybrid_score(self, 
                               vector_similarity: float,
                               graph_context: Dict,
                               expansion_paths: List) -> float:
        """Calculate combined score from vector similarity and graph structure"""
        
        # Base score from vector similarity
        base_score = vector_similarity
        
        # Boost based on graph connectivity (more connections = more important)
        connectivity_boost = min(len(expansion_paths) * 0.05, 0.2)  # Max 20% boost
        
        # Boost based on graph context richness
        context_boost = 0
        if graph_context:
            # Check for plans, categories, requirements, etc.
            context_elements = ['plans', 'categories', 'requirements', 'dependents', 'features']
            available_context = sum(1 for elem in context_elements if elem in graph_context and graph_context[elem])
            context_boost = min(available_context * 0.03, 0.15)  # Max 15% boost
        
        hybrid_score = base_score + connectivity_boost + context_boost
        return min(hybrid_score, 1.0)  # Cap at 1.0
    
    def structural_to_semantic_query(self, 
                                   graph_path: List[str],
                                   semantic_expansion: bool = True) -> Dict[str, Any]:
        """
        Convert structural graph path to semantic understanding
        1. Follow specific graph relationships
        2. Optionally expand with semantically similar items
        3. Return comprehensive results
        """
        
        logger.info(f"🕸️ Structural→Semantic query: {' → '.join(graph_path)}")
        
        if len(graph_path) < 2:
            return {'error': 'Graph path must contain at least 2 nodes'}
        
        # Step 1: Execute graph traversal
        graph_results = self._execute_graph_path(graph_path)
        
        if not graph_results:
            return {'error': 'No results found for graph path'}
        
        # Step 2: Optional semantic expansion
        semantic_results = {}
        if semantic_expansion:
            for result in graph_results[:5]:  # Limit to top 5 for semantic expansion
                item_name = result.get('name', '')
                if item_name:
                    # Find semantically similar items
                    similar_items = self.hybrid_storage.query_hybrid(
                        item_name,
                        n_results=3,
                        similarity_threshold=0.5
                    )
                    semantic_results[item_name] = similar_items
        
        return {
            'graph_path': graph_path,
            'structural_results': graph_results,
            'semantic_expansions': semantic_results,
            'result_count': len(graph_results),
            'processed_at': datetime.now().isoformat()
        }
    
    def _execute_graph_path(self, path: List[str]) -> List[Dict]:
        """Execute a specific graph path query"""
        
        # Build dynamic Cypher query based on path
        if len(path) == 2:
            # Simple relationship query
            query = f"""
            MATCH (start {{name: $start_name}})-[r]-(end {{name: $end_name}})
            RETURN start, end, r, 
                   labels(start)[0] as start_type,
                   labels(end)[0] as end_type,
                   type(r) as relationship_type
            """
            params = {'start_name': path[0], 'end_name': path[1]}
            
        else:
            # Multi-hop path query
            path_length = len(path) - 1
            query = f"""
            MATCH path = (start {{name: $start_name}})-[*1..{path_length}]-(end {{name: $end_name}})
            WHERE all(node in nodes(path)[1..-1] WHERE node.name IN $intermediate_names)
            RETURN path, nodes(path) as path_nodes, relationships(path) as path_relationships
            LIMIT 10
            """
            params = {
                'start_name': path[0],
                'end_name': path[-1],
                'intermediate_names': path[1:-1]
            }
        
        results = []
        try:
            with self.hybrid_storage.neo4j_driver.session() as session:
                cypher_results = session.run(query, params).data()
                
                for result in cypher_results:
                    if 'path' in result:
                        # Multi-hop result
                        path_nodes = result['path_nodes']
                        path_relationships = result['path_relationships']
                        
                        processed_result = {
                            'path_length': len(path_nodes),
                            'nodes': [dict(node.items()) for node in path_nodes],
                            'relationships': [{'type': rel.type, 'properties': dict(rel.items())} 
                                            for rel in path_relationships]
                        }
                    else:
                        # Simple relationship result
                        processed_result = {
                            'start_node': dict(result['start'].items()),
                            'end_node': dict(result['end'].items()),
                            'relationship': {
                                'type': result['relationship_type'],
                                'properties': dict(result['r'].items())
                            },
                            'start_type': result['start_type'],
                            'end_type': result['end_type']
                        }
                    
                    results.append(processed_result)
                    
        except Exception as e:
            logger.error(f"Error executing graph path query: {e}")
        
        return results
    
    def find_semantic_clusters(self, 
                              content_type: str = 'features',
                              similarity_threshold: float = 0.7,
                              min_cluster_size: int = 3) -> Dict[str, List[Dict]]:
        """Find clusters of semantically similar items"""
        
        logger.info(f"🔍 Finding semantic clusters in {content_type}...")
        
        collection_name = f'm365_{content_type}'
        if collection_name not in self.hybrid_storage.collections:
            return {}
        
        collection = self.hybrid_storage.collections[collection_name]
        
        try:
            # Get all items from collection
            all_items = collection.get(
                include=['ids', 'embeddings', 'documents', 'metadatas']
            )
            
            if not all_items['ids']:
                return {}
            
            embeddings = np.array(all_items['embeddings'])
            
            # Calculate pairwise similarities
            similarities = np.dot(embeddings, embeddings.T)
            
            # Find clusters using simple threshold-based clustering
            clusters = {}
            processed_items = set()
            cluster_id = 0
            
            for i, item_id in enumerate(all_items['ids']):
                if item_id in processed_items:
                    continue
                
                # Find similar items
                similar_indices = np.where(similarities[i] >= similarity_threshold)[0]
                
                if len(similar_indices) >= min_cluster_size:
                    cluster_name = f"cluster_{cluster_id}"
                    clusters[cluster_name] = []
                    
                    for idx in similar_indices:
                        similar_id = all_items['ids'][idx]
                        if similar_id not in processed_items:
                            clusters[cluster_name].append({
                                'id': similar_id,
                                'name': all_items['metadatas'][idx].get('name', 'Unknown'),
                                'similarity_to_center': float(similarities[i][idx]),
                                'text': all_items['documents'][idx][:150] + "..."
                            })
                            processed_items.add(similar_id)
                    
                    cluster_id += 1
            
            logger.info(f"✅ Found {len(clusters)} semantic clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Error finding semantic clusters: {e}")
            return {}
    
    def bridge_recommendation_engine(self, 
                                   user_query: str,
                                   recommendation_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Advanced recommendation engine combining vector and graph insights
        """
        
        logger.info(f"🎯 Bridge recommendations for: '{user_query}'")
        
        recommendations = {
            'direct_matches': [],
            'related_features': [],
            'suitable_plans': [],
            'learning_path': [],
            'alternatives': [],
            'metadata': {
                'query': user_query,
                'recommendation_type': recommendation_type,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Step 1: Get semantic matches
        semantic_results = self.semantic_to_structural_query(user_query, expand_depth=2)
        
        # Step 2: Extract direct matches
        if semantic_results.get('primary_matches'):
            recommendations['direct_matches'] = semantic_results['primary_matches'][:5]
        
        # Step 3: Build related features from graph expansions
        seen_features = set()
        for match in semantic_results.get('expanded_matches', []):
            if match['content_type'] == 'features' and match['name'] not in seen_features:
                recommendations['related_features'].append({
                    'name': match['name'],
                    'connection_path': match['relationship_path'],
                    'relevance_score': match['indirect_relevance']
                })
                seen_features.add(match['name'])
        
        # Step 4: Identify suitable plans
        plan_scores = {}
        for match in semantic_results.get('primary_matches', []):
            if match['content_type'] == 'features':
                # Get plans that include this feature
                graph_context = self.hybrid_storage.get_graph_context(match['name'], 'Feature')
                if 'plans' in graph_context:
                    for plan in graph_context['plans']:
                        plan_name = plan.get('name', '')
                        if plan_name:
                            if plan_name not in plan_scores:
                                plan_scores[plan_name] = 0
                            plan_scores[plan_name] += match['hybrid_score']
        
        # Sort plans by score
        sorted_plans = sorted(plan_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations['suitable_plans'] = [
            {'name': plan_name, 'relevance_score': score}
            for plan_name, score in sorted_plans[:5]
        ]
        
        # Step 5: Create learning path (sequence of related features)
        if recommendations['direct_matches']:
            primary_feature = recommendations['direct_matches'][0]['name']
            learning_path = self._generate_learning_path(primary_feature)
            recommendations['learning_path'] = learning_path
        
        # Step 6: Find alternatives using semantic clustering
        if recommendations['direct_matches']:
            primary_match = recommendations['direct_matches'][0]
            alternatives = self._find_alternatives(primary_match, user_query)
            recommendations['alternatives'] = alternatives
        
        return recommendations
    
    def _generate_learning_path(self, start_feature: str, max_depth: int = 4) -> List[Dict]:
        """Generate a learning path starting from a feature"""
        
        learning_path = []
        
        with self.hybrid_storage.neo4j_driver.session() as session:
            # Find prerequisite chain
            prereq_query = """
            MATCH path = (start:Feature {name: $feature_name})<-[:REQUIRES*0..3]-(prereq:Feature)
            WHERE start <> prereq
            RETURN prereq, length(path) as depth
            ORDER BY depth DESC
            """
            
            try:
                prereq_results = session.run(prereq_query, feature_name=start_feature).data()
                
                # Add prerequisites first (foundational learning)
                for result in prereq_results:
                    prereq = dict(result['prereq'].items())
                    learning_path.append({
                        'name': prereq.get('name', ''),
                        'type': 'prerequisite',
                        'depth': result['depth'],
                        'description': prereq.get('description', '')
                    })
                
                # Add the main feature
                main_feature = session.run(
                    "MATCH (f:Feature {name: $name}) RETURN f", 
                    name=start_feature
                ).single()
                
                if main_feature:
                    feature_data = dict(main_feature['f'].items())
                    learning_path.append({
                        'name': start_feature,
                        'type': 'primary',
                        'depth': 0,
                        'description': feature_data.get('description', '')
                    })
                
                # Add dependent features (advanced learning)
                dependent_query = """
                MATCH path = (start:Feature {name: $feature_name})-[:REQUIRES*0..2]->(dependent:Feature)
                WHERE start <> dependent
                RETURN dependent, length(path) as depth
                ORDER BY depth ASC
                """
                
                dependent_results = session.run(dependent_query, feature_name=start_feature).data()
                
                for result in dependent_results:
                    dependent = dict(result['dependent'].items())
                    learning_path.append({
                        'name': dependent.get('name', ''),
                        'type': 'dependent',
                        'depth': result['depth'],
                        'description': dependent.get('description', '')
                    })
                
            except Exception as e:
                logger.debug(f"Error generating learning path: {e}")
        
        return learning_path[:max_depth]
    
    def _find_alternatives(self, primary_match: Dict, original_query: str) -> List[Dict]:
        """Find alternative features/solutions"""
        
        alternatives = []
        
        # Get features in same category
        if primary_match.get('metadata', {}).get('category'):
            category = primary_match['metadata']['category']
            
            similar_in_category = self.hybrid_storage.query_hybrid(
                f"{category} features",
                content_types=['features'],
                n_results=5,
                similarity_threshold=0.3
            )
            
            for item in similar_in_category.get('features', []):
                if item['metadata']['name'] != primary_match['name']:
                    alternatives.append({
                        'name': item['metadata']['name'],
                        'similarity': item['similarity'],
                        'reason': f"Similar {category} feature",
                        'text': item['text'][:100] + "..."
                    })
        
        # Get semantically similar features
        semantic_alternatives = self.hybrid_storage.query_hybrid(
            original_query,
            content_types=['features'],
            n_results=10,
            similarity_threshold=0.4
        )
        
        for item in semantic_alternatives.get('features', []):
            if (item['metadata']['name'] != primary_match['name'] and 
                not any(alt['name'] == item['metadata']['name'] for alt in alternatives)):
                
                alternatives.append({
                    'name': item['metadata']['name'],
                    'similarity': item['similarity'],
                    'reason': "Semantically similar",
                    'text': item['text'][:100] + "..."
                })
        
        # Sort by similarity and limit
        return sorted(alternatives, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def export_bridge_insights(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Export comprehensive bridge insights and analytics"""
        
        if not output_file:
            output_file = f"../data/bridge_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"📊 Exporting bridge insights to: {output_file}")
        
        insights = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'vector_graph_mappings': len(self.vector_graph_mapping.get('vector_to_graph', {}).get('features', {})),
                'total_collections': len(self.hybrid_storage.collections)
            },
            'vector_statistics': {},
            'graph_statistics': {},
            'bridge_statistics': {},
            'sample_queries': []
        }
        
        # Get storage statistics
        storage_stats = self.hybrid_storage.get_storage_statistics()
        insights['vector_statistics'] = storage_stats.get('vector_storage', {})
        insights['graph_statistics'] = storage_stats.get('graph_storage', {})
        insights['bridge_statistics'] = storage_stats.get('hybrid_metrics', {})
        
        # Add sample successful queries
        sample_queries = [
            "email security features",
            "collaboration tools",
            "data protection",
            "SharePoint capabilities"
        ]
        
        for query in sample_queries:
            try:
                result = self.semantic_to_structural_query(query, max_results=3)
                insights['sample_queries'].append({
                    'query': query,
                    'primary_results': len(result.get('primary_matches', [])),
                    'expanded_results': len(result.get('expanded_matches', [])),
                    'success': True
                })
            except Exception as e:
                insights['sample_queries'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        logger.info(f"✅ Bridge insights exported successfully")
        return insights
    
    def test_bridge_functionality(self):
        """Comprehensive test of bridge functionality"""
        
        logger.info("🧪 Testing Vector-Graph Bridge functionality...")
        
        test_results = {
            'semantic_to_structural': [],
            'structural_to_semantic': [],
            'recommendations': [],
            'clustering': []
        }
        
        # Test 1: Semantic to Structural
        logger.info("  🔍 Testing semantic→structural queries...")
        semantic_queries = [
            "security and compliance features",
            "team collaboration tools",
            "document management"
        ]
        
        for query in semantic_queries:
            try:
                result = self.semantic_to_structural_query(query, max_results=5)
                test_results['semantic_to_structural'].append({
                    'query': query,
                    'primary_matches': len(result.get('primary_matches', [])),
                    'expanded_matches': len(result.get('expanded_matches', [])),
                    'success': True
                })
                logger.info(f"    ✅ '{query}': {len(result.get('primary_matches', []))} primary matches")
            except Exception as e:
                test_results['semantic_to_structural'].append({
                    'query': query,
                    'error': str(e),
                    'success': False
                })
                logger.warning(f"    ⚠️ '{query}' failed: {e}")
        
        # Test 2: Structural to Semantic  
        logger.info("  🕸️ Testing structural→semantic queries...")
        try:
            # This would need actual node names from your graph
            sample_path = ["Microsoft Teams", "Microsoft 365 Business Premium"]  # Example path
            result = self.structural_to_semantic_query(sample_path)
            test_results['structural_to_semantic'].append({
                'path': sample_path,
                'results': len(result.get('structural_results', [])),
                'success': True
            })
            logger.info(f"    ✅ Path query successful")
        except Exception as e:
            test_results['structural_to_semantic'].append({
                'error': str(e),
                'success': False
            })
            logger.warning(f"    ⚠️ Structural query failed: {e}")
        
        # Test 3: Recommendation Engine
        logger.info("  🎯 Testing recommendation engine...")
        try:
            recommendations = self.bridge_recommendation_engine("email security")
            test_results['recommendations'].append({
                'query': "email security",
                'direct_matches': len(recommendations.get('direct_matches', [])),
                'related_features': len(recommendations.get('related_features', [])),
                'suitable_plans': len(recommendations.get('suitable_plans', [])),
                'success': True
            })
            logger.info(f"    ✅ Recommendations generated successfully")
        except Exception as e:
            test_results['recommendations'].append({
                'error': str(e),
                'success': False
            })
            logger.warning(f"    ⚠️ Recommendations failed: {e}")
        
        # Test 4: Semantic Clustering
        logger.info("  🔍 Testing semantic clustering...")
        try:
            clusters = self.find_semantic_clusters('features', similarity_threshold=0.6)
            test_results['clustering'].append({
                'content_type': 'features',
                'clusters_found': len(clusters),
                'success': True
            })
            logger.info(f"    ✅ Found {len(clusters)} semantic clusters")
        except Exception as e:
            test_results['clustering'].append({
                'error': str(e),
                'success': False
            })
            logger.warning(f"    ⚠️ Clustering failed: {e}")
        
        logger.info("✅ Bridge functionality testing complete")
        return test_results

def setup_vector_graph_bridge(hybrid_storage: HybridStorage) -> VectorGraphBridge:
    """Main setup function for vector-graph bridge"""
    
    logger.info("🚀 PHASE 2.4: VECTOR-GRAPH BRIDGE SETUP")
    logger.info("="*50)
    
    try:
        # Initialize bridge
        bridge = VectorGraphBridge(hybrid_storage)
        
        # Test bridge functionality
        test_results = bridge.test_bridge_functionality()
        
        # Export insights
        insights = bridge.export_bridge_insights()
        
        # Display results
        logger.info("\n🌉 BRIDGE STATISTICS")
        logger.info("-" * 30)
        logger.info(f"Vector-Graph Mappings: {insights['metadata']['vector_graph_mappings']}")
        logger.info(f"Total Collections: {insights['metadata']['total_collections']}")
        
        success_count = sum(1 for test_type in test_results.values() 
                          for test in test_type if test.get('success', False))
        total_tests = sum(len(tests) for tests in test_results.values())
        
        logger.info(f"Test Success Rate: {success_count}/{total_tests}")
        
        logger.info("\n✅ PHASE 2.4 COMPLETE!")
        logger.info("="*50)
        
        return bridge
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 2.4: {e}")
        raise

if __name__ == "__main__":
    # This would typically be called after setting up hybrid storage
    from .03_hybrid_storage import setup_hybrid_storage
    
    # Setup hybrid storage first
    hybrid_storage = setup_hybrid_storage()
    
    # Setup bridge
    bridge = setup_vector_graph_bridge(hybrid_storage)
    
    # Display final status
    print("\n🎯 Vector-Graph Bridge Ready!")
    print("✅ Phase 2 Complete - Ready for Phase 3: Query Processing")