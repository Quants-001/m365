# phase2_hybrid_storage/05_advanced_nlp_wrapper.py
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Core NLP libraries with fallbacks
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - using pattern-based NLP fallback")

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import our hybrid storage components
from phase2_03_hybrid_storage import HybridStorage
from phase2_04_vector_graph_bridge import VectorGraphBridge

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """User intent classification"""
    GREETING = "greeting"
    QUESTION = "question"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    FEATURE_LOOKUP = "feature_lookup"
    PLAN_INQUIRY = "plan_inquiry"
    PRICING = "pricing"
    AVAILABILITY = "availability"
    REQUIREMENTS = "requirements"
    GOODBYE = "goodbye"
    CLARIFICATION = "clarification"
    UNKNOWN = "unknown"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"          # Single entity lookup
    MODERATE = "moderate"      # Multiple entities or relationships
    COMPLEX = "complex"        # Multi-hop reasoning or comparisons
    CONVERSATIONAL = "conversational"  # Greetings, clarifications

class EntityType(Enum):
    """Recognized entity types"""
    FEATURE = "feature"
    PLAN = "plan"
    CATEGORY = "category"
    PRICE = "price"
    REQUIREMENT = "requirement"
    PERSON = "person"
    ORGANIZATION = "organization"
    TIME = "time"

@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    normalized_form: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ProcessedQuery:
    """Complete query analysis result"""
    original_query: str
    intent: IntentType
    complexity: QueryComplexity
    entities: List[ExtractedEntity]
    expanded_query: str
    keywords: List[str]
    search_strategy: str
    filters: Dict[str, Any]
    confidence: float
    processing_metadata: Dict[str, Any]

class AdvancedNLPWrapper:
    """
    Advanced NLP Wrapper for Microsoft 365 Query Processing
    Integrates with hybrid storage and provides sophisticated query understanding
    """
    
    def __init__(self, 
                 hybrid_storage: Optional[HybridStorage] = None,
                 vector_graph_bridge: Optional[VectorGraphBridge] = None,
                 language_model: str = "en_core_web_sm"):
        
        self.hybrid_storage = hybrid_storage
        self.vector_graph_bridge = vector_graph_bridge
        
        # Initialize NLP components
        self.nlp_model = None
        self.sentence_transformer = None
        
        # Load spaCy model with fallback
        if SPACY_AVAILABLE:
            try:
                self.nlp_model = spacy.load(language_model)
                logger.info(f"✅ Loaded spaCy model: {language_model}")
            except OSError:
                logger.warning(f"⚠️ Could not load {language_model}, trying basic model...")
                try:
                    self.nlp_model = spacy.load("en_core_web_sm")
                    logger.info("✅ Loaded basic spaCy model")
                except OSError:
                    logger.warning("⚠️ No spaCy models available, using pattern-based fallback")
                    self.nlp_model = None
        
        # Load sentence transformer for semantic understanding
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded sentence transformer for semantic analysis")
            except Exception as e:
                logger.warning(f"⚠️ Could not load sentence transformer: {e}")
        
        # Load knowledge bases
        self._load_knowledge_bases()
        
        # Initialize conversation context
        self.conversation_history = []
        self.context_entities = set()
        
        logger.info("🧠 Advanced NLP Wrapper initialized")
    
    def _load_knowledge_bases(self):
        """Load Microsoft 365 specific knowledge bases"""
        
        # Greeting patterns
        self.greeting_patterns = {
            'formal': ['hello', 'good morning', 'good afternoon', 'good evening', 'greetings'],
            'informal': ['hi', 'hey', 'yo', 'sup', 'what\'s up'],
            'questions': ['how are you', 'how do you do', 'what\'s new'],
            'introductions': ['i am', 'my name is', 'i\'m', 'call me']
        }
        
        # Goodbye patterns
        self.goodbye_patterns = [
            'goodbye', 'bye', 'see you', 'farewell', 'take care', 'later',
            'thanks', 'thank you', 'that\'s all', 'that\'s it', 'done'
        ]
        
        # Microsoft 365 specific entities
        self.m365_features = [
            'teams', 'sharepoint', 'onedrive', 'outlook', 'exchange', 'word', 'excel', 
            'powerpoint', 'onenote', 'power bi', 'power apps', 'power automate',
            'yammer', 'stream', 'planner', 'to do', 'forms', 'sway', 'delve',
            'advanced threat protection', 'atp', 'defender', 'compliance center',
            'information protection', 'dlp', 'data loss prevention', 'sensitivity labels'
        ]
        
        self.m365_plans = [
            'business basic', 'business standard', 'business premium',
            'enterprise e1', 'enterprise e3', 'enterprise e5', 'e1', 'e3', 'e5',
            'microsoft 365 personal', 'microsoft 365 family',
            'office 365 e1', 'office 365 e3', 'office 365 e5'
        ]
        
        # Intent keywords mapping
        self.intent_keywords = {
            IntentType.COMPARISON: ['compare', 'vs', 'versus', 'difference', 'better', 'best', 'which'],
            IntentType.RECOMMENDATION: ['recommend', 'suggest', 'should', 'need', 'help me choose', 'what would'],
            IntentType.FEATURE_LOOKUP: ['features', 'includes', 'has', 'contains', 'capabilities'],
            IntentType.PLAN_INQUIRY: ['plan', 'subscription', 'license', 'tier'],
            IntentType.PRICING: ['price', 'cost', 'expensive', 'cheap', 'budget', 'pay'],
            IntentType.AVAILABILITY: ['available', 'included', 'comes with', 'part of'],
            IntentType.REQUIREMENTS: ['require', 'need', 'prerequisite', 'depends on']
        }
        
        logger.info("📚 Knowledge bases loaded")
    
    def detect_greeting(self, text: str) -> Tuple[bool, str, float]:
        """
        Enhanced greeting detection with confidence scoring
        Returns: (is_greeting, greeting_type, confidence)
        """
        text_lower = text.lower().strip()
        
        # Check for exact greeting matches
        for greeting_type, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Calculate confidence based on pattern match quality
                    if text_lower.startswith(pattern):
                        confidence = 0.9
                    elif pattern == text_lower:
                        confidence = 1.0
                    else:
                        confidence = 0.7
                    
                    return True, greeting_type, confidence
        
        # Check for greeting-like sentence patterns
        greeting_indicators = [
            r'^(hi|hello|hey)\s+(there|everyone|team)?',
            r'^good\s+(morning|afternoon|evening|day)',
            r'^(how\s+are\s+you|how\s+do\s+you\s+do)',
            r'^(nice\s+to\s+meet\s+you|pleased\s+to\s+meet\s+you)',
        ]
        
        for pattern in greeting_indicators:
            if re.search(pattern, text_lower):
                return True, 'contextual', 0.8
        
        # Check if it's a short, friendly phrase
        if len(text.split()) <= 3 and any(word in text_lower for word in ['hi', 'hello', 'hey']):
            return True, 'informal', 0.6
        
        return False, 'none', 0.0
    
    def detect_goodbye(self, text: str) -> Tuple[bool, float]:
        """
        Detect goodbye/farewell intents
        Returns: (is_goodbye, confidence)
        """
        text_lower = text.lower().strip()
        
        for pattern in self.goodbye_patterns:
            if pattern in text_lower:
                if text_lower.startswith(pattern) or text_lower == pattern:
                    return True, 0.9
                else:
                    return True, 0.7
        
        # Pattern-based goodbye detection
        goodbye_patterns = [
            r'^(bye|goodbye|farewell)',
            r'(thank\s+you|thanks).*(?:bye|goodbye|that\'s\s+all)',
            r'^(see\s+you|talk\s+to\s+you)',
            r'(that\'s\s+it|that\'s\s+all|done|finished)'
        ]
        
        for pattern in goodbye_patterns:
            if re.search(pattern, text_lower):
                return True, 0.8
        
        return False, 0.0
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy and pattern matching"""
        
        entities = []
        
        # Use spaCy if available
        if self.nlp_model:
            doc = self.nlp_model(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_entity_type(ent.label_)
                if entity_type:
                    entities.append(ExtractedEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        confidence=0.8,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        metadata={'spacy_label': ent.label_}
                    ))
        
        # Pattern-based entity extraction for M365 specifics
        entities.extend(self._extract_m365_entities(text))
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types"""
        
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'DATE': EntityType.TIME,
            'TIME': EntityType.TIME,
            'MONEY': EntityType.PRICE,
            'PRODUCT': EntityType.FEATURE
        }
        
        return mapping.get(spacy_label)
    
    def _extract_m365_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract Microsoft 365 specific entities using patterns"""
        
        entities = []
        text_lower = text.lower()
        
        # Extract M365 features
        for feature in self.m365_features:
            pattern = r'\b' + re.escape(feature.lower()) + r'\b'
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                entities.append(ExtractedEntity(
                    text=feature,
                    entity_type=EntityType.FEATURE,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=feature,
                    metadata={'source': 'pattern_matching', 'category': 'feature'}
                ))
        
        # Extract M365 plans
        for plan in self.m365_plans:
            pattern = r'\b' + re.escape(plan.lower()) + r'\b'
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                entities.append(ExtractedEntity(
                    text=plan,
                    entity_type=EntityType.PLAN,
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    normalized_form=plan,
                    metadata={'source': 'pattern_matching', 'category': 'plan'}
                ))
        
        # Extract price patterns
        price_patterns = [
            r'\$\d+(?:\.\d{2})?(?:\s*(?:per|/)\s*(?:month|user|year))?',
            r'\d+(?:\.\d{2})?\s*(?:dollars?|usd)(?:\s*(?:per|/)\s*(?:month|user|year))?'
        ]
        
        for pattern in price_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=EntityType.PRICE,
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={'source': 'price_pattern'}
                ))
        
        return entities
    
    def classify_intent(self, text: str, entities: List[ExtractedEntity]) -> Tuple[IntentType, float]:
        """Classify user intent based on text and extracted entities"""
        
        text_lower = text.lower()
        
        # Check for greeting first
        is_greeting, _, greeting_confidence = self.detect_greeting(text)
        if is_greeting:
            return IntentType.GREETING, greeting_confidence
        
        # Check for goodbye
        is_goodbye, goodbye_confidence = self.detect_goodbye(text)
        if is_goodbye:
            return IntentType.GOODBYE, goodbye_confidence
        
        # Intent classification based on keywords and patterns
        intent_scores = {}
        
        for intent_type, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight based on position and exact match
                    if text_lower.startswith(keyword):
                        score += 3
                    elif f" {keyword} " in f" {text_lower} ":
                        score += 2
                    else:
                        score += 1
            
            if score > 0:
                intent_scores[intent_type] = score / len(keywords)
        
        # Adjust scores based on entities
        plan_entities = [e for e in entities if e.entity_type == EntityType.PLAN]
        feature_entities = [e for e in entities if e.entity_type == EntityType.FEATURE]
        price_entities = [e for e in entities if e.entity_type == EntityType.PRICE]
        
        if len(plan_entities) >= 2:
            intent_scores[IntentType.COMPARISON] = intent_scores.get(IntentType.COMPARISON, 0) + 0.5
        
        if plan_entities and feature_entities:
            intent_scores[IntentType.AVAILABILITY] = intent_scores.get(IntentType.AVAILABILITY, 0) + 0.3
        
        if price_entities:
            intent_scores[IntentType.PRICING] = intent_scores.get(IntentType.PRICING, 0) + 0.4
        
        # Question patterns
        question_patterns = [
            r'^(what|how|when|where|why|which|who)',
            r'^(is|are|do|does|can|could|would|will)',
            r'^(show\s+me|tell\s+me|explain)',
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                intent_scores[IntentType.QUESTION] = intent_scores.get(IntentType.QUESTION, 0) + 0.3
                break
        
        # Return highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = min(best_intent[1], 1.0)  # Cap at 1.0
            return best_intent[0], confidence
        
        # Default to question if no clear intent
        return IntentType.QUESTION, 0.5
    
    def determine_complexity(self, text: str, entities: List[ExtractedEntity], intent: IntentType) -> QueryComplexity:
        """Determine query complexity level"""
        
        # Conversational intents are always conversational complexity
        if intent in [IntentType.GREETING, IntentType.GOODBYE, IntentType.CLARIFICATION]:
            return QueryComplexity.CONVERSATIONAL
        
        # Count different entity types
        entity_types = set(e.entity_type for e in entities)
        entity_count = len(entities)
        
        # Check for comparison indicators
        comparison_words = ['vs', 'versus', 'compare', 'difference', 'better', 'best']
        has_comparison = any(word in text.lower() for word in comparison_words)
        
        # Check for multi-part questions
        multi_part_indicators = ['and', 'also', 'what about', 'how about', 'additionally']
        has_multiple_parts = any(indicator in text.lower() for indicator in multi_part_indicators)
        
        # Complex queries
        if (has_comparison or has_multiple_parts or 
            len(entity_types) >= 3 or entity_count >= 4 or
            intent == IntentType.COMPARISON):
            return QueryComplexity.COMPLEX
        
        # Moderate queries
        if (len(entity_types) >= 2 or entity_count >= 2 or
            intent in [IntentType.RECOMMENDATION, IntentType.REQUIREMENTS]):
            return QueryComplexity.MODERATE
        
        # Simple queries
        return QueryComplexity.SIMPLE
    
    def expand_query(self, text: str, entities: List[ExtractedEntity], intent: IntentType) -> str:
        """Expand query with context and synonyms"""
        
        expanded_parts = [text]
        
        # Add context based on entities
        for entity in entities:
            if entity.entity_type == EntityType.FEATURE and entity.normalized_form:
                # Add Microsoft 365 context
                expanded_parts.append(f"Microsoft 365 {entity.normalized_form}")
            
            elif entity.entity_type == EntityType.PLAN and entity.normalized_form:
                # Add plan context
                expanded_parts.append(f"Microsoft 365 {entity.normalized_form} subscription")
        
        # Add intent-specific context
        if intent == IntentType.FEATURE_LOOKUP:
            expanded_parts.append("features capabilities functionality")
        elif intent == IntentType.PRICING:
            expanded_parts.append("cost price subscription pricing")
        elif intent == IntentType.COMPARISON:
            expanded_parts.append("compare difference features plans")
        
        # Join and clean up
        expanded_query = " ".join(expanded_parts)
        
        # Remove duplicates and excessive whitespace
        words = expanded_query.lower().split()
        unique_words = []
        seen = set()
        
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        return " ".join(unique_words)
    
    def determine_search_strategy(self, intent: IntentType, complexity: QueryComplexity, entities: List[ExtractedEntity]) -> str:
        """Determine optimal search strategy for the query"""
        
        # Strategy mapping based on intent and complexity
        if intent in [IntentType.GREETING, IntentType.GOODBYE]:
            return "conversational"  # Special handling for conversational queries
        
        if intent == IntentType.COMPARISON and complexity == QueryComplexity.COMPLEX:
            return "hybrid_expansion"
        
        if intent == IntentType.FEATURE_LOOKUP and len(entities) == 1:
            return "semantic_first"
        
        if intent in [IntentType.AVAILABILITY, IntentType.REQUIREMENTS]:
            return "graph_first"
        
        if complexity == QueryComplexity.COMPLEX:
            return "contextual_rerank"
        
        if complexity == QueryComplexity.SIMPLE:
            return "semantic_first"
        
        # Default strategy
        return "hybrid_expansion"
    
    def extract_filters(self, entities: List[ExtractedEntity], intent: IntentType) -> Dict[str, Any]:
        """Extract search filters from entities and intent"""
        
        filters = {
            'plan_filter': None,
            'feature_filter': None,
            'content_type_priority': [],
            'keyword_filters': [],
            'price_range': None
        }
        
        # Extract entity-based filters
        for entity in entities:
            if entity.entity_type == EntityType.PLAN and entity.normalized_form:
                filters['plan_filter'] = entity.normalized_form
            
            elif entity.entity_type == EntityType.FEATURE and entity.normalized_form:
                filters['feature_filter'] = entity.normalized_form
                filters['keyword_filters'].append(entity.normalized_form)
        
        # Set content type priorities based on intent
        if intent == IntentType.PLAN_INQUIRY:
            filters['content_type_priority'] = ['plans', 'features', 'relationships']
        elif intent == IntentType.FEATURE_LOOKUP:
            filters['content_type_priority'] = ['features', 'relationships', 'plans']
        elif intent == IntentType.COMPARISON:
            filters['content_type_priority'] = ['plans', 'features']
        else:
            filters['content_type_priority'] = ['features', 'plans', 'relationships']
        
        return filters
    
    def generate_greeting_response(self, greeting_type: str) -> Dict[str, Any]:
        """Generate appropriate greeting response"""
        
        responses = {
            'formal': "Hello! I'm your Microsoft 365 assistant. How can I help you today?",
            'informal': "Hi there! I'm here to help you with Microsoft 365 questions. What would you like to know?",
            'questions': "I'm doing well, thank you for asking! I'm here to help you with Microsoft 365. What can I assist you with?",
            'introductions': "Nice to meet you! I'm your Microsoft 365 assistant, ready to help with any questions about plans, features, or capabilities.",
            'contextual': "Hello! I'm here to help you navigate Microsoft 365. What would you like to explore today?"
        }
        
        response_text = responses.get(greeting_type, responses['formal'])
        
        return {
            'text': response_text,
            'type': 'greeting',
            'confidence': 1.0,
            'suggestions': [
                "What features are included in Microsoft 365?",
                "Compare Business plans",
                "Show me security features",
                "What's the difference between E3 and E5?"
            ]
        }
    
    def generate_goodbye_response(self) -> Dict[str, Any]:
        """Generate appropriate goodbye response"""
        
        responses = [
            "Thank you for using Microsoft 365 assistant. Have a great day!",
            "Goodbye! Feel free to ask me anything about Microsoft 365 anytime.",
            "Take care! I'm here whenever you need help with Microsoft 365.",
            "Thanks for the conversation! Don't hesitate to reach out with more questions."
        ]
        
        import random
        response_text = random.choice(responses)
        
        return {
            'text': response_text,
            'type': 'goodbye',
            'confidence': 1.0,
            'final': True
        }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ProcessedQuery:
        """
        Main query processing method
        Analyzes query and returns comprehensive processing results
        """
        
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing query: '{query[:100]}...'")
        
        # Extract entities
        entities = self.extract_entities(query)
        
        # Classify intent
        intent, intent_confidence = self.classify_intent(query, entities)
        
        # Determine complexity
        complexity = self.determine_complexity(query, entities, intent)
        
        # Expand query for better search
        expanded_query = self.expand_query(query, entities, intent)
        
        # Determine search strategy
        search_strategy = self.determine_search_strategy(intent, complexity, entities)
        
        # Extract filters
        filters = self.extract_filters(entities, intent)
        
        # Extract keywords
        keywords = self._extract_keywords(query, entities)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(intent_confidence, entities, complexity)
        
        # Processing metadata
        processing_metadata = {
            'processing_time_ms': int((datetime.now() - start_time).total_seconds() * 1000),
            'spacy_available': self.nlp_model is not None,
            'transformers_available': self.sentence_transformer is not None,
            'entity_count': len(entities),
            'context_provided': context is not None
        }
        
        result = ProcessedQuery(
            original_query=query,
            intent=intent,
            complexity=complexity,
            entities=entities,
            expanded_query=expanded_query,
            keywords=keywords,
            search_strategy=search_strategy,
            filters=filters,
            confidence=overall_confidence,
            processing_metadata=processing_metadata
        )
        
        logger.info(f"✅ Query processed: intent={intent.value}, complexity={complexity.value}, strategy={search_strategy}")
        
        return result
    
    def _extract_keywords(self, query: str, entities: List[ExtractedEntity]) -> List[str]:
        """Extract important keywords from query"""
        
        # Remove stop words and extract meaningful terms
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'what', 'how', 'when', 'where', 'why', 'which', 'who', 'whom', 'whose'
        }
        
        # Extract words from query
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add entity normalized forms
        for entity in entities:
            if entity.normalized_form:
                entity_words = entity.normalized_form.lower().split()
                keywords.extend([word for word in entity_words if word not in stop_words])
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:10]  # Limit to top 10 keywords
    
    def _calculate_confidence(self, intent_confidence: float, entities: List[ExtractedEntity], complexity: QueryComplexity) -> float:
        """Calculate overall confidence in query processing"""
        
        # Base confidence from intent classification
        base_confidence = intent_confidence
        
        # Boost confidence based on entity extraction
        entity_boost = min(0.2, len(entities) * 0.05)
        
        # Adjust based on complexity (simpler queries are more confident)
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MODERATE: 0.9,
            QueryComplexity.COMPLEX: 0.8,
            QueryComplexity.CONVERSATIONAL: 1.1
        }
        
        final_confidence = (base_confidence + entity_boost) * complexity_multiplier[complexity]
        
        return min(final_confidence, 1.0)  # Cap at 1.0
    
    def update_conversation_context(self, processed_query: ProcessedQuery, response: Optional[Dict[str, Any]] = None):
        """Update conversation context for better follow-up processing"""
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': processed_query.original_query,
            'intent': processed_query.intent.value,
            'entities': [e.text for e in processed_query.entities],
            'response_provided': response is not None
        })
        
        # Update context entities
        for entity in processed_query.entities:
            if entity.entity_type in [EntityType.FEATURE, EntityType.PLAN]:
                self.context_entities.add(entity.normalized_form or entity.text)
        
        # Limit history size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        # Limit context entities
        if len(self.context_entities) > 10:
            # Keep most recent entities (simplified approach)
            recent_entities = set()
            for entry in self.conversation_history[-5:]:
                recent_entities.update(entry.get('entities', []))
            self.context_entities = recent_entities
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state"""
        
        return {
            'total_queries': len(self.conversation_history),
            'context_entities': list(self.context_entities),
            'recent_intents': [entry['intent'] for entry in self.conversation_history[-5:]],
            'nlp_capabilities': {
                'spacy_available': self.nlp_model is not None,
                'transformers_available': self.sentence_transformer is not None,
                'knowledge_base_loaded': True
            }
        }

def setup_advanced_nlp_wrapper(hybrid_storage: Optional[HybridStorage] = None,
                              vector_graph_bridge: Optional[VectorGraphBridge] = None) -> AdvancedNLPWrapper:
    """Main setup function for Advanced NLP Wrapper"""
    
    logger.info("🚀 PHASE 2.5: ADVANCED NLP WRAPPER SETUP")
    logger.info("="*50)
    
    try:
        # Initialize NLP wrapper
        nlp_wrapper = AdvancedNLPWrapper(
            hybrid_storage=hybrid_storage,
            vector_graph_bridge=vector_graph_bridge
        )
        
        # Test with sample queries
        test_queries = [
            "Hello there!",
            "What security features are in E3?",
            "Compare Business Basic vs Business Standard",
            "Thanks, that's all I need",
            "Is Advanced Threat Protection available in Enterprise E5?",
            "Show me collaboration tools for small teams"
        ]
        
        logger.info("\n🧪 TESTING NLP PROCESSING")
        logger.info("-" * 40)
        
        for query in test_queries:
            processed = nlp_wrapper.process_query(query)
            
            logger.info(f"Query: '{query}'")
            logger.info(f"  Intent: {processed.intent.value}")
            logger.info(f"  Complexity: {processed.complexity.value}")
            logger.info(f"  Strategy: {processed.search_strategy}")
            logger.info(f"  Entities: {len(processed.entities)}")
            logger.info(f"  Confidence: {processed.confidence:.2f}")
            
            # Test greeting/goodbye responses
            if processed.intent == IntentType.GREETING:
                is_greeting, greeting_type, confidence = nlp_wrapper.detect_greeting(query)
                response = nlp_wrapper.generate_greeting_response(greeting_type)
                logger.info(f"  Greeting Response: '{response['text'][:60]}...'")
            
            elif processed.intent == IntentType.GOODBYE:
                response = nlp_wrapper.generate_goodbye_response()
                logger.info(f"  Goodbye Response: '{response['text'][:60]}...'")
            
            logger.info("")
        
        # Display capabilities
        summary = nlp_wrapper.get_conversation_summary()
        
        logger.info("🧠 NLP CAPABILITIES")
        logger.info("-" * 30)
        logger.info(f"spaCy Available: {summary['nlp_capabilities']['spacy_available']}")
        logger.info(f"Transformers Available: {summary['nlp_capabilities']['transformers_available']}")
        logger.info(f"Knowledge Base: {summary['nlp_capabilities']['knowledge_base_loaded']}")
        logger.info(f"M365 Features: {len(nlp_wrapper.m365_features)}")
        logger.info(f"M365 Plans: {len(nlp_wrapper.m365_plans)}")
        
        logger.info("\n✅ PHASE 2.5 COMPLETE!")
        logger.info("="*50)
        
        return nlp_wrapper
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 2.5: {e}")
        raise


class ConversationalInterface:
    """
    Conversational Interface integrating NLP processing with Hybrid Storage
    Provides natural language interaction with Microsoft 365 knowledge base
    """
    
    def __init__(self, 
                 hybrid_storage: HybridStorage,
                 vector_graph_bridge: VectorGraphBridge,
                 nlp_wrapper: AdvancedNLPWrapper):
        
        self.hybrid_storage = hybrid_storage
        self.vector_graph_bridge = vector_graph_bridge
        self.nlp_wrapper = nlp_wrapper
        
        # Conversation state
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_active = True
        
        logger.info(f"💬 Conversational Interface initialized (Session: {self.session_id})")
    
    def process_conversation(self, user_input: str) -> Dict[str, Any]:
        """
        Main conversation processing method
        Handles the complete flow from user input to response
        """
        
        logger.info(f"💬 Processing conversation: '{user_input[:50]}...'")
        
        # Step 1: Process query with NLP
        processed_query = self.nlp_wrapper.process_query(user_input)
        
        # Step 2: Handle special intents
        if processed_query.intent == IntentType.GREETING:
            is_greeting, greeting_type, confidence = self.nlp_wrapper.detect_greeting(user_input)
            response = self.nlp_wrapper.generate_greeting_response(greeting_type)
            
            return {
                'response': response,
                'processed_query': processed_query,
                'search_results': None,
                'conversation_active': True
            }
        
        elif processed_query.intent == IntentType.GOODBYE:
            response = self.nlp_wrapper.generate_goodbye_response()
            self.conversation_active = False
            
            return {
                'response': response,
                'processed_query': processed_query,
                'search_results': None,
                'conversation_active': False
            }
        
        # Step 3: Execute search strategy
        search_results = self._execute_search(processed_query)
        
        # Step 4: Generate response
        response = self._generate_response(processed_query, search_results)
        
        # Step 5: Update conversation context
        self.nlp_wrapper.update_conversation_context(processed_query, response)
        
        return {
            'response': response,
            'processed_query': processed_query,
            'search_results': search_results,
            'conversation_active': self.conversation_active
        }
    
    def _execute_search(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """Execute the appropriate search strategy"""
        
        strategy = processed_query.search_strategy
        query = processed_query.expanded_query
        filters = processed_query.filters
        
        if strategy == "conversational":
            # No search needed for conversational queries
            return {'results': [], 'strategy': strategy}
        
        elif strategy == "semantic_first":
            return self.vector_graph_bridge.semantic_to_structural_query(
                query, 
                expand_depth=1,
                max_results=10
            )
        
        elif strategy == "graph_first":
            # Use graph search with pattern matching
            with self.hybrid_storage.neo4j_driver.session() as session:
                # Simple feature/plan lookup
                if filters.get('feature_filter'):
                    graph_query = """
                    MATCH (f:Feature)
                    WHERE toLower(f.name) CONTAINS toLower($feature)
                    OPTIONAL MATCH (f)-[:AVAILABLE_IN]->(p:Plan)
                    OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
                    RETURN f.name as name, f.description as description,
                           collect(DISTINCT p.name) as plans,
                           c.name as category
                    LIMIT 10
                    """
                    results = session.run(graph_query, feature=filters['feature_filter']).data()
                    
                elif filters.get('plan_filter'):
                    graph_query = """
                    MATCH (p:Plan)
                    WHERE toLower(p.name) CONTAINS toLower($plan)
                    OPTIONAL MATCH (f:Feature)-[:AVAILABLE_IN]->(p)
                    RETURN p.name as name, p.type as type,
                           collect(DISTINCT f.name) as features
                    LIMIT 10
                    """
                    results = session.run(graph_query, plan=filters['plan_filter']).data()
                
                else:
                    # Fallback to fulltext search
                    results = []
                
                return {
                    'results': results,
                    'strategy': strategy,
                    'source': 'neo4j_direct'
                }
        
        elif strategy == "hybrid_expansion":
            return self.vector_graph_bridge.semantic_to_structural_query(
                query,
                expand_depth=2,
                max_results=15
            )
        
        elif strategy == "contextual_rerank":
            # Use the bridge's recommendation engine
            return self.vector_graph_bridge.bridge_recommendation_engine(
                query,
                recommendation_type='comprehensive'
            )
        
        else:
            # Default fallback
            return self.hybrid_storage.query_hybrid(
                query,
                content_types=filters.get('content_type_priority', ['features', 'plans']),
                n_results=10
            )
    
    def _generate_response(self, processed_query: ProcessedQuery, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate natural language response based on search results"""
        
        intent = processed_query.intent
        complexity = processed_query.complexity
        entities = processed_query.entities
        
        # Extract relevant results
        primary_results = search_results.get('primary_matches', search_results.get('results', []))
        expanded_results = search_results.get('expanded_matches', [])
        
        # Generate response based on intent
        if intent == IntentType.FEATURE_LOOKUP:
            return self._generate_feature_response(processed_query, primary_results)
        
        elif intent == IntentType.PLAN_INQUIRY:
            return self._generate_plan_response(processed_query, primary_results)
        
        elif intent == IntentType.COMPARISON:
            return self._generate_comparison_response(processed_query, primary_results)
        
        elif intent == IntentType.AVAILABILITY:
            return self._generate_availability_response(processed_query, primary_results)
        
        elif intent == IntentType.RECOMMENDATION:
            return self._generate_recommendation_response(processed_query, search_results)
        
        else:
            return self._generate_general_response(processed_query, primary_results)
    
    def _generate_feature_response(self, processed_query: ProcessedQuery, results: List[Dict]) -> Dict[str, Any]:
        """Generate response for feature lookup queries"""
        
        if not results:
            return {
                'text': "I couldn't find specific information about that feature. Could you provide more details or try rephrasing your question?",
                'type': 'no_results',
                'confidence': 0.3
            }
        
        # Extract feature entities from query
        feature_entities = [e for e in processed_query.entities if e.entity_type == EntityType.FEATURE]
        feature_name = feature_entities[0].text if feature_entities else "the requested feature"
        
        top_result = results[0]
        response_text = f"Here's what I found about {feature_name}:\n\n"
        
        if hasattr(top_result, 'get') and top_result.get('text'):
            response_text += top_result['text'][:300]
            if len(top_result['text']) > 300:
                response_text += "..."
        else:
            # Handle different result formats
            name = top_result.get('name', feature_name)
            description = top_result.get('description', '')
            plans = top_result.get('plans', [])
            
            response_text += f"{name}"
            if description:
                response_text += f": {description}"
            
            if plans:
                response_text += f"\n\nAvailable in: {', '.join(plans[:5])}"
                if len(plans) > 5:
                    response_text += f" and {len(plans) - 5} more plans"
        
        # Add related suggestions
        suggestions = []
        if len(results) > 1:
            for result in results[1:4]:  # Next 3 results
                name = result.get('name') or result.get('metadata', {}).get('name', 'Related item')
                suggestions.append(f"Tell me about {name}")
        
        return {
            'text': response_text,
            'type': 'feature_info',
            'confidence': 0.8,
            'suggestions': suggestions
        }
    
    def _generate_plan_response(self, processed_query: ProcessedQuery, results: List[Dict]) -> Dict[str, Any]:
        """Generate response for plan inquiry queries"""
        
        if not results:
            return {
                'text': "I couldn't find information about that plan. Could you specify which Microsoft 365 plan you're interested in?",
                'type': 'no_results',
                'confidence': 0.3
            }
        
        plan_entities = [e for e in processed_query.entities if e.entity_type == EntityType.PLAN]
        plan_name = plan_entities[0].text if plan_entities else "the requested plan"
        
        top_result = results[0]
        response_text = f"Here's information about {plan_name}:\n\n"
        
        # Extract plan details
        if hasattr(top_result, 'get'):
            features = top_result.get('features', [])
            plan_type = top_result.get('type', '')
            feature_count = top_result.get('feature_count', 0)
            
            if plan_type:
                response_text += f"Type: {plan_type}\n"
            
            if feature_count:
                response_text += f"Includes {feature_count} features\n"
            
            if features:
                response_text += f"\nKey features:\n"
                for feature in features[:8]:  # Show top 8 features
                    response_text += f"• {feature}\n"
                
                if len(features) > 8:
                    response_text += f"• ... and {len(features) - 8} more features"
        
        return {
            'text': response_text,
            'type': 'plan_info',
            'confidence': 0.8,
            'suggestions': [
                f"Compare {plan_name} with other plans",
                "What are the pricing options?",
                "Show me security features"
            ]
        }
    
    def _generate_comparison_response(self, processed_query: ProcessedQuery, results: List[Dict]) -> Dict[str, Any]:
        """Generate response for comparison queries"""
        
        plan_entities = [e for e in processed_query.entities if e.entity_type == EntityType.PLAN]
        
        if len(plan_entities) < 2:
            return {
                'text': "To help you compare plans, please specify which Microsoft 365 plans you'd like to compare. For example: 'Compare Business Basic vs Business Standard' or 'What's the difference between E3 and E5?'",
                'type': 'clarification_needed',
                'confidence': 0.6
            }
        
        plan1_name = plan_entities[0].text
        plan2_name = plan_entities[1].text
        
        response_text = f"Here's a comparison between {plan1_name} and {plan2_name}:\n\n"
        
        # Find results for each plan
        plan1_results = [r for r in results if plan1_name.lower() in str(r).lower()]
        plan2_results = [r for r in results if plan2_name.lower() in str(r).lower()]
        
        if plan1_results:
            response_text += f"**{plan1_name}:**\n"
            result = plan1_results[0]
            if hasattr(result, 'get') and result.get('text'):
                response_text += result['text'][:200] + "...\n\n"
        
        if plan2_results:
            response_text += f"**{plan2_name}:**\n"
            result = plan2_results[0]
            if hasattr(result, 'get') and result.get('text'):
                response_text += result['text'][:200] + "...\n\n"
        
        if not (plan1_results or plan2_results):
            response_text += "I found some general information that might help with your comparison:\n\n"
            for result in results[:2]:
                text = result.get('text', str(result))[:150]
                response_text += f"• {text}...\n"
        
        return {
            'text': response_text,
            'type': 'comparison',
            'confidence': 0.7,
            'suggestions': [
                "Show me pricing for these plans",
                "Which plan is better for small business?",
                "Compare security features"
            ]
        }
    
    def _generate_availability_response(self, processed_query: ProcessedQuery, results: List[Dict]) -> Dict[str, Any]:
        """Generate response for availability queries"""
        
        feature_entities = [e for e in processed_query.entities if e.entity_type == EntityType.FEATURE]
        plan_entities = [e for e in processed_query.entities if e.entity_type == EntityType.PLAN]
        
        if feature_entities and plan_entities:
            feature_name = feature_entities[0].text
            plan_name = plan_entities[0].text
            
            # Look for specific availability information
            availability_found = False
            for result in results:
                result_text = str(result).lower()
                if feature_name.lower() in result_text and plan_name.lower() in result_text:
                    availability_found = True
                    if 'available' in result_text or 'included' in result_text:
                        response_text = f"Yes, {feature_name} is available in {plan_name}."
                    else:
                        response_text = f"Based on the information I found, {feature_name} appears to be associated with {plan_name}."
                    break
            
            if not availability_found:
                response_text = f"I couldn't find specific information about {feature_name} availability in {plan_name}. "
                response_text += "Let me show you what I found about these items:\n\n"
                
                for result in results[:2]:
                    text = result.get('text', str(result))[:150]
                    response_text += f"• {text}...\n"
        
        else:
            response_text = "To check availability, please specify both the feature and plan you're interested in. "
            response_text += "For example: 'Is Advanced Threat Protection available in Enterprise E3?'"
        
        return {
            'text': response_text,
            'type': 'availability',
            'confidence': 0.7
        }
    
    def _generate_recommendation_response(self, processed_query: ProcessedQuery, search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for recommendation queries"""
        
        # Extract recommendation data from bridge results
        direct_matches = search_results.get('direct_matches', [])
        suitable_plans = search_results.get('suitable_plans', [])
        related_features = search_results.get('related_features', [])
        
        response_text = "Based on your requirements, here are my recommendations:\n\n"
        
        if suitable_plans:
            response_text += "**Recommended Plans:**\n"
            for plan in suitable_plans[:3]:
                name = plan.get('name', 'Unknown Plan')
                score = plan.get('relevance_score', 0)
                response_text += f"• {name} (relevance: {score:.1f})\n"
            response_text += "\n"
        
        if related_features:
            response_text += "**Key Features to Consider:**\n"
            for feature in related_features[:5]:
                name = feature.get('name', 'Unknown Feature')
                response_text += f"• {name}\n"
            response_text += "\n"
        
        if direct_matches:
            response_text += "**Additional Information:**\n"
            for match in direct_matches[:2]:
                text = match.get('text', '')[:150]
                response_text += f"• {text}...\n"
        
        return {
            'text': response_text,
            'type': 'recommendation',
            'confidence': 0.8,
            'suggestions': [
                "Tell me more about the recommended plans",
                "Compare the top recommendations",
                "What's the pricing for these options?"
            ]
        }
    
    def _generate_general_response(self, processed_query: ProcessedQuery, results: List[Dict]) -> Dict[str, Any]:
        """Generate general response for other query types"""
        
        if not results:
            return {
                'text': "I couldn't find specific information for your query. Could you try rephrasing your question or providing more details about what you're looking for?",
                'type': 'no_results',
                'confidence': 0.3,
                'suggestions': [
                    "Ask about Microsoft 365 features",
                    "Compare business plans",
                    "Show me security capabilities"
                ]
            }
        
        response_text = "Here's what I found:\n\n"
        
        # Show top results
        for i, result in enumerate(results[:3], 1):
            text = result.get('text', str(result))
            name = result.get('name') or result.get('metadata', {}).get('name', f'Result {i}')
            
            response_text += f"**{name}:**\n"
            response_text += text[:200]
            if len(text) > 200:
                response_text += "..."
            response_text += "\n\n"
        
        return {
            'text': response_text,
            'type': 'general_info',
            'confidence': 0.6
        }


if __name__ == "__main__":
    # Example usage and testing
    from phase2_03_hybrid_storage import setup_hybrid_storage
    from phase2_04_vector_graph_bridge import setup_vector_graph_bridge
    
    # Setup components
    hybrid_storage = setup_hybrid_storage()
    vector_graph_bridge = setup_vector_graph_bridge(hybrid_storage)
    nlp_wrapper = setup_advanced_nlp_wrapper(hybrid_storage, vector_graph_bridge)
    
    # Create conversational interface
    conv_interface = ConversationalInterface(
        hybrid_storage, 
        vector_graph_bridge, 
        nlp_wrapper
    )
    
    # Test conversation
    test_queries = [
        "Hello!",
        "What security features are in Enterprise E3?",
        "Compare Business Basic and Business Standard",
        "Thanks, that's all!"
    ]
    
    print("🚀 CONVERSATIONAL INTERFACE DEMO")
    print("="*50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        result = conv_interface.process_conversation(query)
        print(f"Assistant: {result['response']['text']}")
        
        if not result['conversation_active']:
            break
    
    print("\n✅ Demo complete!")