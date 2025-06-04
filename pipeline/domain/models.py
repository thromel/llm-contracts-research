"""Enhanced data models with auto-validation and MongoDB integration."""

from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, computed_field
import hashlib
import uuid

from pipeline.infrastructure.database import MongoDocument
from pipeline.foundation.types import PipelineStage as FoundationPipelineStage


class Platform(str, Enum):
    """Source platform enumeration."""
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"


class ContractType(str, Enum):
    """LLM Contract types from the taxonomy."""
    DATA_TYPE = "data_type"
    OUTPUT_FORMAT = "output_format"
    RATE_LIMIT = "rate_limit"
    CONTEXT_LENGTH = "context_length"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    MAX_TOKENS = "max_tokens"
    JSON_SCHEMA = "json_schema"
    FUNCTION_CALLING = "function_calling"
    STREAM_FORMAT = "stream_format"
    SAFETY_FILTERS = "safety_filters"


class PipelineStage(str, Enum):
    """Pipeline stages from Khairunnesa et al."""
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_CREATION = "model_creation"
    MODEL_TRAINING = "model_training"
    MODEL_TUNING = "model_tuning"
    MODEL_TESTING = "model_testing"
    MODEL_DEPLOYMENT = "model_deployment"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"


class RootCause(str, Enum):
    """Root cause categories."""
    UNACCEPTABLE_INPUT = "unacceptable_input"
    MISSING_CALL = "missing_call"
    IMPROPER_USAGE = "improper_usage"
    VERSION_MISMATCH = "version_mismatch"
    CONFIG_ERROR = "config_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CONTEXT_OVERFLOW = "context_overflow"
    FORMAT_VIOLATION = "format_violation"
    DEPENDENCY_ISSUE = "dependency_issue"
    API_DEPRECATION = "api_deprecation"
    AUTHENTICATION_ERROR = "authentication_error"


class Effect(str, Enum):
    """Effect categories from Khairunnesa Table 5."""
    CRASH = "crash"
    HANG = "hang"
    BAD_PERF = "bad_perf"  # Bad Performance
    DC = "dc"  # Data Corruption
    IF = "if"  # Incorrect Functionality
    MOB = "mob"  # Memory Out of Bounds
    TIMEOUT = "timeout"
    DEGRADED_OUTPUT = "degraded_output"
    UNKNOWN = "unknown"


class RawPost(MongoDocument):
    """Raw post data from acquisition stage with enhanced validation."""
    
    _collection_name = "raw_posts"
    _indexes = [
        {"keys": [("platform", 1), ("source_id", 1)], "name": "platform_source_unique", "unique": True},
        {"keys": [("content_hash", 1)], "name": "content_hash_unique", "unique": True, "sparse": True},
        {"keys": [("acquisition_timestamp", -1)], "name": "acquisition_timestamp_desc"},
        {"keys": [("tags", 1)], "name": "tags_multikey"},
        {"keys": [("score", -1)], "name": "score_desc"}
    ]
    
    # Core fields
    platform: Platform
    source_id: str = Field(..., min_length=1, description="GitHub issue ID or SO post ID")
    url: str = Field(..., description="Full URL to the post")
    title: str = Field(..., min_length=1, description="Post title")
    body_md: str = Field(..., description="Markdown content of the post")
    author: str = Field(..., description="Post author username")
    
    # Timestamps
    post_created_at: datetime = Field(..., description="When the post was originally created")
    post_updated_at: Optional[datetime] = Field(default=None, description="When the post was last updated")
    
    # Scoring and engagement
    score: int = Field(default=0, description="Post score (upvotes - downvotes)")
    answer_score: Optional[int] = Field(default=None, description="Score of accepted answer (SO only)")
    view_count: int = Field(default=0, description="Number of views")
    comments_count: int = Field(default=0, ge=0, description="Number of comments")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Post tags")
    labels: List[str] = Field(default_factory=list, description="GitHub labels")
    
    # Platform-specific fields
    state: Optional[str] = Field(default=None, description="GitHub: open/closed, SO: answered/unanswered")
    accepted_answer_id: Optional[str] = Field(default=None, description="SO: ID of accepted answer")
    
    # Provenance and processing
    acquisition_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When acquired by our system")
    acquisition_version: str = Field(default="2.0.0", description="Version of acquisition system")
    content_hash: Optional[str] = Field(default=None, description="Hash for deduplication")
    
    @computed_field
    @property
    def computed_content_hash(self) -> str:
        """Compute content hash for deduplication."""
        content = f"{self.title}|||{self.body_md}|||{self.platform.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    @field_validator('platform')
    @classmethod
    def validate_platform_consistency(cls, v: Platform) -> Platform:
        """Validate platform enum."""
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization processing."""
        super().model_post_init(__context)
        
        # Auto-generate content hash if not provided
        if not self.content_hash:
            self.content_hash = self.computed_content_hash
        
        # Generate ID based on platform and source_id if not provided
        if not self.id:
            self.id = f"{self.platform.value}_{self.source_id}"
    
    def get_display_summary(self) -> str:
        """Get a human-readable summary for logging/display."""
        return f"{self.platform.value.title()} {self.source_id}: {self.title[:50]}..."


class FilteredPost(MongoDocument):
    """Post after keyword pre-filtering with confidence scoring."""
    
    _collection_name = "filtered_posts"
    _indexes = [
        {"keys": [("raw_post_id", 1)], "name": "raw_post_id_unique", "unique": True},
        {"keys": [("passed_keyword_filter", 1)], "name": "passed_filter"},
        {"keys": [("filter_confidence", -1)], "name": "confidence_desc"},
        {"keys": [("filter_timestamp", -1)], "name": "filter_timestamp_desc"},
        {"keys": [("matched_keywords", 1)], "name": "matched_keywords_multikey"}
    ]
    
    raw_post_id: str = Field(..., description="Reference to RawPost ID")
    
    # Filter results
    passed_keyword_filter: bool = Field(default=False, description="Whether post passed filtering")
    filter_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score [0-1]")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords that matched")
    
    # Extracted content for LLM screening
    relevant_snippets: List[str] = Field(default_factory=list, description="Relevant text snippets")
    potential_contracts: List[str] = Field(default_factory=list, description="Potential contract violations identified")
    
    # Quality metrics
    noise_indicators: List[str] = Field(default_factory=list, description="Signals indicating noisy content")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall content quality")
    
    # Processing metadata
    filter_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When filtering was performed")
    filter_version: str = Field(default="2.0.0", description="Version of filtering system")
    processing_time_ms: Optional[int] = Field(default=None, ge=0, description="Processing time in milliseconds")
    
    @field_validator('filter_confidence', 'quality_score')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Validate score ranges."""
        return max(0.0, min(1.0, v))
    
    def should_screen_with_llm(self, threshold: float = 0.3) -> bool:
        """Determine if post should proceed to LLM screening."""
        return self.passed_keyword_filter and self.filter_confidence >= threshold


class LLMScreeningResult(BaseModel):
    """Result from LLM screening with enhanced classification."""
    
    # Core decision
    decision: str = Field(..., pattern="^(Y|N|Unsure)$", description="Final decision: Y/N/Unsure")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision [0-1]")
    rationale: str = Field(..., min_length=10, description="Detailed reasoning for decision")
    
    # Model information
    model_used: str = Field(..., description="LLM model identifier")
    model_provider: str = Field(..., description="Provider (openai, anthropic, etc.)")
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = Field(default=None, ge=0)
    
    # Enhanced classification
    contract_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed violation analysis")
    pipeline_stage: Optional[PipelineStage] = Field(default=None, description="Affected pipeline stage")
    root_cause: Optional[RootCause] = Field(default=None, description="Identified root cause")
    effect: Optional[Effect] = Field(default=None, description="Observed effect")
    
    # Research value assessment
    novel_patterns: Optional[str] = Field(default=None, description="Novel patterns identified")
    research_value: Optional[str] = Field(default=None, description="Research value assessment")
    verification_notes: Optional[str] = Field(default=None, description="Additional verification notes")
    
    # Quality indicators
    response_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality of LLM response")
    requires_human_review: bool = Field(default=False, description="Whether human review is needed")
    
    @field_validator('decision')
    @classmethod
    def validate_decision(cls, v: str) -> str:
        """Validate decision format."""
        if v not in ('Y', 'N', 'Unsure'):
            raise ValueError("Decision must be Y, N, or Unsure")
        return v
    
    def is_positive(self) -> bool:
        """Check if this is a positive classification."""
        return self.decision == "Y"
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if confidence exceeds threshold."""
        return self.confidence >= threshold


class HumanLabel(BaseModel):
    """Single human label with enhanced validation."""
    
    rater_id: str = Field(..., min_length=1, description="Unique rater identifier")
    
    # Core classification
    is_contract_violation: bool = Field(..., description="Whether this represents a contract violation")
    contract_type: Optional[ContractType] = Field(default=None, description="Type of contract violated")
    pipeline_stage: Optional[PipelineStage] = Field(default=None, description="Affected pipeline stage")
    root_cause: Optional[RootCause] = Field(default=None, description="Root cause of issue")
    effect: Optional[Effect] = Field(default=None, description="Observed effect")
    
    # Detailed assessment
    notes: str = Field(default="", description="Detailed notes from rater")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Rater confidence [0-1]")
    difficulty: int = Field(default=3, ge=1, le=5, description="Labeling difficulty [1-5]")
    
    # Quality indicators
    clear_evidence: bool = Field(default=True, description="Whether evidence is clear")
    requires_domain_knowledge: bool = Field(default=False, description="Whether domain expertise needed")
    ambiguous_case: bool = Field(default=False, description="Whether case is ambiguous")
    
    # Provenance
    labeling_timestamp: datetime = Field(default_factory=datetime.utcnow)
    labeling_session_id: Optional[str] = Field(default=None, description="Session this label belongs to")
    time_spent_seconds: Optional[int] = Field(default=None, ge=0, description="Time spent labeling")
    
    @field_validator('rater_id')
    @classmethod
    def validate_rater_id(cls, v: str) -> str:
        """Validate rater ID format."""
        if not v.startswith(('R', 'r')):
            v = f"R{v}"
        return v.upper()
    
    def get_label_summary(self) -> Dict[str, Any]:
        """Get summary of label for analysis."""
        return {
            "rater_id": self.rater_id,
            "is_violation": self.is_contract_violation,
            "contract_type": self.contract_type.value if self.contract_type else None,
            "confidence": self.confidence,
            "difficulty": self.difficulty
        }


class LabelledPost(MongoDocument):
    """Post with complete labeling from multiple raters."""
    
    _collection_name = "labelled_posts"
    _indexes = [
        {"keys": [("filtered_post_id", 1)], "name": "filtered_post_id_unique", "unique": True},
        {"keys": [("labelling_session_id", 1)], "name": "session_id"},
        {"keys": [("final_decision", 1)], "name": "final_decision"},
        {"keys": [("majority_agreement", 1)], "name": "majority_agreement"},
        {"keys": [("labelling_timestamp", -1)], "name": "labelling_timestamp_desc"}
    ]
    
    filtered_post_id: str = Field(..., description="Reference to FilteredPost ID")
    
    # LLM screening results
    bulk_screening: Optional[LLMScreeningResult] = Field(default=None, description="Bulk screening result")
    borderline_screening: Optional[LLMScreeningResult] = Field(default=None, description="Borderline screening result")
    agentic_screening: Optional[LLMScreeningResult] = Field(default=None, description="Agentic screening result")
    
    # Human labels
    human_labels: List[HumanLabel] = Field(default_factory=list, description="All human labels")
    
    # Consensus analysis
    final_decision: Optional[bool] = Field(default=None, description="Final consensus decision")
    majority_agreement: bool = Field(default=False, description="Whether majority agreed")
    required_arbitration: bool = Field(default=False, description="Whether arbitration was needed")
    consensus_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in consensus")
    
    # Agreement metrics
    fleiss_kappa: Optional[float] = Field(default=None, description="Fleiss kappa for this item")
    pairwise_agreements: Dict[str, float] = Field(default_factory=dict, description="Pairwise agreement scores")
    
    # Processing metadata
    labelling_session_id: str = Field(default="", description="Session ID for this labeling")
    labelling_timestamp: datetime = Field(default_factory=datetime.utcnow)
    labelling_version: str = Field(default="2.0.0", description="Version of labeling system")
    
    @computed_field
    @property
    def rater_count(self) -> int:
        """Number of human raters."""
        return len(self.human_labels)
    
    @computed_field
    @property
    def positive_labels(self) -> int:
        """Number of positive labels."""
        return sum(1 for label in self.human_labels if label.is_contract_violation)
    
    def calculate_agreement(self) -> Dict[str, Any]:
        """Calculate agreement metrics for this post."""
        if len(self.human_labels) < 2:
            return {"insufficient_labels": True}
        
        positive_count = self.positive_labels
        total_count = self.rater_count
        
        # Simple agreement percentage
        majority_threshold = total_count / 2
        agreement_pct = max(positive_count, total_count - positive_count) / total_count
        
        return {
            "total_raters": total_count,
            "positive_labels": positive_count,
            "negative_labels": total_count - positive_count,
            "agreement_percentage": agreement_pct,
            "has_majority": positive_count > majority_threshold or (total_count - positive_count) > majority_threshold,
            "is_unanimous": positive_count == 0 or positive_count == total_count
        }
    
    def get_final_label(self) -> Optional[bool]:
        """Get the final consensus label."""
        if self.final_decision is not None:
            return self.final_decision
        
        # Calculate based on majority
        agreement = self.calculate_agreement()
        if agreement.get("has_majority", False):
            return self.positive_labels > (self.rater_count / 2)
        
        return None  # No clear majority


class LabellingSession(MongoDocument):
    """Metadata for a labeling session with progress tracking."""
    
    _collection_name = "labelling_sessions"
    _indexes = [
        {"keys": [("session_date", -1)], "name": "session_date_desc"},
        {"keys": [("status", 1)], "name": "status"},
        {"keys": [("moderator", 1)], "name": "moderator"}
    ]
    
    session_name: str = Field(..., min_length=1, description="Human-readable session name")
    session_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Participants
    raters: List[str] = Field(default_factory=list, description="List of rater IDs")
    moderator: str = Field(..., description="Session moderator")
    
    # Progress tracking
    posts_assigned: int = Field(default=0, ge=0, description="Total posts assigned")
    posts_completed: int = Field(default=0, ge=0, description="Posts completed by all raters")
    posts_in_progress: int = Field(default=0, ge=0, description="Posts currently being labeled")
    
    # Status management
    status: str = Field(default="active", pattern="^(active|completed|paused|cancelled)$")
    completion_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall completion %")
    
    # Quality metrics
    target_fleiss_kappa: float = Field(default=0.80, ge=0.0, le=1.0, description="Target agreement threshold")
    current_fleiss_kappa: Optional[float] = Field(default=None, description="Current measured agreement")
    quality_passed: bool = Field(default=False, description="Whether quality threshold met")
    
    # Timing
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    actual_completion: Optional[datetime] = Field(default=None, description="Actual completion time")
    
    @field_validator('posts_completed')
    @classmethod
    def validate_completion_progress(cls, v: int, info) -> int:
        """Ensure completed doesn't exceed assigned."""
        if 'posts_assigned' in info.data and v > info.data['posts_assigned']:
            raise ValueError("Completed posts cannot exceed assigned posts")
        return v
    
    def update_progress(self, completed: int, in_progress: int = 0) -> None:
        """Update session progress."""
        self.posts_completed = completed
        self.posts_in_progress = in_progress
        
        if self.posts_assigned > 0:
            self.completion_percentage = (completed / self.posts_assigned) * 100
        
        # Auto-update status
        if completed >= self.posts_assigned:
            self.status = "completed"
            self.actual_completion = datetime.utcnow()
        
        self.mark_updated()


class ReliabilityMetrics(MongoDocument):
    """Fleiss Kappa and reliability metrics with detailed analysis."""
    
    _collection_name = "reliability_metrics"
    _indexes = [
        {"keys": [("session_id", 1)], "name": "session_id"},
        {"keys": [("calculation_date", -1)], "name": "calculation_date_desc"},
        {"keys": [("fleiss_kappa", -1)], "name": "fleiss_kappa_desc"},
        {"keys": [("passes_threshold", 1)], "name": "passes_threshold"}
    ]
    
    session_id: str = Field(..., description="Associated labeling session")
    calculation_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Core Fleiss Kappa metrics
    fleiss_kappa: float = Field(..., description="Fleiss kappa coefficient")
    kappa_std_error: float = Field(default=0.0, ge=0.0, description="Standard error of kappa")
    kappa_confidence_interval: Tuple[float, float] = Field(default=(0.0, 0.0), description="95% confidence interval")
    
    # Sample characteristics
    n_items: int = Field(..., gt=0, description="Number of items rated")
    n_raters: int = Field(..., gt=0, description="Number of raters")
    n_categories: int = Field(..., gt=0, description="Number of categories")
    
    # Agreement breakdown
    observed_agreement: float = Field(..., ge=0.0, le=1.0, description="Observed proportion agreement")
    expected_agreement: float = Field(..., ge=0.0, le=1.0, description="Expected agreement by chance")
    
    # Detailed analysis
    category_agreement: Dict[str, float] = Field(default_factory=dict, description="Per-category agreement")
    confusion_matrix: Dict[str, Any] = Field(default_factory=dict, description="Confusion matrix data")
    rater_statistics: Dict[str, Any] = Field(default_factory=dict, description="Per-rater statistics")
    
    # Quality assessment
    quality_threshold: float = Field(default=0.80, description="Threshold for acceptable agreement")
    passes_threshold: bool = Field(default=False, description="Whether kappa meets threshold")
    needs_review: bool = Field(default=False, description="Whether results need review")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Statistical significance
    z_score: Optional[float] = Field(default=None, description="Z-score for significance testing")
    p_value: Optional[float] = Field(default=None, description="P-value for significance")
    is_significant: bool = Field(default=False, description="Whether result is statistically significant")
    
    def interpret_kappa(self) -> str:
        """Provide interpretation of kappa value according to Landis & Koch."""
        if self.fleiss_kappa < 0:
            return "Poor (less than chance agreement)"
        elif self.fleiss_kappa < 0.20:
            return "Slight"
        elif self.fleiss_kappa < 0.40:
            return "Fair"
        elif self.fleiss_kappa < 0.60:
            return "Moderate"
        elif self.fleiss_kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality metrics."""
        return {
            "fleiss_kappa": self.fleiss_kappa,
            "interpretation": self.interpret_kappa(),
            "passes_threshold": self.passes_threshold,
            "n_items": self.n_items,
            "n_raters": self.n_raters,
            "observed_agreement": self.observed_agreement
        }


class TaxonomyDefinition(MongoDocument):
    """Comprehensive taxonomy definition with versioning."""
    
    _collection_name = "taxonomy_definitions"
    _indexes = [
        {"keys": [("name", 1), ("version", 1)], "name": "name_version_unique", "unique": True},
        {"keys": [("created_date", -1)], "name": "created_date_desc"},
        {"keys": [("is_active", 1)], "name": "is_active"}
    ]
    
    name: str = Field(..., min_length=1, description="Taxonomy name")
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$", description="Semantic version")
    description: str = Field(default="", description="Taxonomy description")
    is_active: bool = Field(default=True, description="Whether this version is active")
    
    # Hierarchical definitions
    contract_types: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Contract type definitions")
    pipeline_stages: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Pipeline stage definitions")
    root_causes: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Root cause definitions")
    effects: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Effect definitions")
    
    # Relationships and mappings
    type_stage_mapping: Dict[str, List[str]] = Field(default_factory=dict, description="Contract type to stage mapping")
    cause_effect_mapping: Dict[str, List[str]] = Field(default_factory=dict, description="Cause to effect mapping")
    severity_mapping: Dict[str, int] = Field(default_factory=dict, description="Effect severity scores")
    
    # Validation rules
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules for taxonomy")
    examples: Dict[str, List[str]] = Field(default_factory=dict, description="Examples for each category")
    
    # Metadata
    created_by: str = Field(..., description="Creator of this taxonomy version")
    review_status: str = Field(default="draft", pattern="^(draft|review|approved|deprecated)$")
    change_summary: str = Field(default="", description="Summary of changes from previous version")
    
    @field_validator('version')
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("Version must be in format X.Y.Z where X, Y, Z are integers")
        return v
    
    def get_category_count(self) -> Dict[str, int]:
        """Get count of items in each category."""
        return {
            "contract_types": len(self.contract_types),
            "pipeline_stages": len(self.pipeline_stages),
            "root_causes": len(self.root_causes),
            "effects": len(self.effects)
        }
    
    def validate_taxonomy_completeness(self) -> List[str]:
        """Validate taxonomy completeness and return issues."""
        issues = []
        
        if not self.contract_types:
            issues.append("No contract types defined")
        if not self.pipeline_stages:
            issues.append("No pipeline stages defined")
        if not self.root_causes:
            issues.append("No root causes defined")
        if not self.effects:
            issues.append("No effects defined")
        
        # Check for orphaned mappings
        for contract_type, stages in self.type_stage_mapping.items():
            if contract_type not in self.contract_types:
                issues.append(f"Contract type '{contract_type}' in mapping but not defined")
            for stage in stages:
                if stage not in self.pipeline_stages:
                    issues.append(f"Pipeline stage '{stage}' in mapping but not defined")
        
        return issues