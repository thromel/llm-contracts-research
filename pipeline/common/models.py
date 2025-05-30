"""
Data models for the LLM Contracts Research Pipeline.

All models support full provenance tracking (raw → filtered → labelled)
as required by the methodology.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import uuid


class Platform(Enum):
    """Source platform enumeration."""
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"


class ContractType(Enum):
    """LLM Contract types from the taxonomy."""
    DATA_TYPE = "data_type"
    OUTPUT_FORMAT = "output_format"
    RATE_LIMIT = "rate_limit"
    CONTEXT_LENGTH = "context_length"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    MAX_TOKENS = "max_tokens"
    JSON_SCHEMA = "json_schema"
    # Add more as defined in taxonomy


class PipelineStage(Enum):
    """Pipeline stages from Khairunnesa et al."""
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_CREATION = "model_creation"
    MODEL_TRAINING = "model_training"
    MODEL_TUNING = "model_tuning"
    MODEL_TESTING = "model_testing"
    MODEL_DEPLOYMENT = "model_deployment"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"


class RootCause(Enum):
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


class Effect(Enum):
    """Effect categories from Khairunnesa Table 5."""
    CRASH = "crash"
    HANG = "hang"
    BAD_PERF = "bad_perf"  # Bad Performance
    DC = "dc"  # Data Corruption
    IF = "if"  # Incorrect Functionality
    MOB = "mob"  # Memory Out of Bounds
    UNKNOWN = "unknown"


@dataclass
class RawPost:
    """Raw post data from acquisition stage."""
    _id: Optional[str] = None
    platform: Platform = Platform.GITHUB
    source_id: str = ""  # GitHub issue ID or SO post ID
    url: str = ""
    title: str = ""
    body_md: str = ""  # Markdown content
    created_at: datetime = field(default_factory=lambda: datetime.now())
    updated_at: Optional[datetime] = None
    score: int = 0
    answer_score: Optional[int] = None  # For SO answers
    tags: List[str] = field(default_factory=list)
    author: str = ""

    # GitHub specific
    state: Optional[str] = None  # open/closed
    labels: List[str] = field(default_factory=list)
    comments_count: int = 0

    # Stack Overflow specific
    accepted_answer_id: Optional[str] = None
    view_count: int = 0

    # Provenance
    acquisition_timestamp: datetime = field(
        default_factory=lambda: datetime.now())
    acquisition_version: str = "1.0.0"

    # Deduplication
    content_hash: Optional[str] = None

    def __post_init__(self):
        """Generate unique ID if not provided."""
        if self._id is None:
            # Generate a unique ID based on platform and source_id
            if self.source_id:
                self._id = f"{self.platform.value}_{self.source_id}"
            else:
                self._id = f"{self.platform.value}_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        data = {
            'platform': self.platform.value,
            'source_id': self.source_id,
            'url': self.url,
            'title': self.title,
            'body_md': self.body_md,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'score': self.score,
            'answer_score': self.answer_score,
            'tags': self.tags,
            'author': self.author,
            'state': self.state,
            'labels': self.labels,
            'comments_count': self.comments_count,
            'accepted_answer_id': self.accepted_answer_id,
            'view_count': self.view_count,
            'acquisition_timestamp': self.acquisition_timestamp,
            'acquisition_version': self.acquisition_version,
            'content_hash': self.content_hash
        }

        # Only include _id if it's not None
        if self._id is not None:
            data['_id'] = self._id

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RawPost':
        """Create RawPost from dictionary (e.g., from MongoDB)."""
        # Handle platform enum conversion
        platform = Platform(data.get('platform', 'github'))

        # Handle datetime fields
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(
                created_at.replace('Z', '+00:00'))
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(
                updated_at.replace('Z', '+00:00'))

        acquisition_timestamp = data.get('acquisition_timestamp')
        if isinstance(acquisition_timestamp, str):
            acquisition_timestamp = datetime.fromisoformat(
                acquisition_timestamp.replace('Z', '+00:00'))
        elif acquisition_timestamp is None:
            acquisition_timestamp = datetime.now()

        return cls(
            _id=data.get('_id'),
            platform=platform,
            source_id=data.get('source_id', ''),
            url=data.get('url', ''),
            title=data.get('title', ''),
            body_md=data.get('body_md', ''),
            created_at=created_at,
            updated_at=updated_at,
            score=data.get('score', 0),
            answer_score=data.get('answer_score'),
            tags=data.get('tags', []),
            author=data.get('author', ''),
            state=data.get('state'),
            labels=data.get('labels', []),
            comments_count=data.get('comments_count', 0),
            accepted_answer_id=data.get('accepted_answer_id'),
            view_count=data.get('view_count', 0),
            acquisition_timestamp=acquisition_timestamp,
            acquisition_version=data.get('acquisition_version', '1.0.0'),
            content_hash=data.get('content_hash')
        )


@dataclass
class FilteredPost:
    """Post after keyword pre-filtering."""
    raw_post_id: str
    _id: Optional[str] = None

    # Filter results
    passed_keyword_filter: bool = False
    matched_keywords: List[str] = field(default_factory=list)
    filter_confidence: float = 0.0

    # Extracted content
    relevant_snippets: List[str] = field(default_factory=list)
    potential_contracts: List[str] = field(default_factory=list)

    # Provenance
    filter_timestamp: datetime = field(default_factory=datetime.utcnow)
    filter_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            '_id': self._id,
            'raw_post_id': self.raw_post_id,
            'passed_keyword_filter': self.passed_keyword_filter,
            'matched_keywords': self.matched_keywords,
            'filter_confidence': self.filter_confidence,
            'relevant_snippets': self.relevant_snippets,
            'potential_contracts': self.potential_contracts,
            'filter_timestamp': self.filter_timestamp,
            'filter_version': self.filter_version
        }


@dataclass
class LLMScreeningResult:
    """Result from LLM screening pass."""
    decision: str  # "Y", "N", "Unsure"
    rationale: str
    confidence: float
    model_used: str  # "deepseek-r1", "gpt-4.1"

    # Contract violation classification (optional)
    contract_violations: Optional[List[Dict[str, Any]]] = None
    novel_patterns: Optional[str] = None
    research_value: Optional[str] = None
    verification_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'decision': self.decision,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'model_used': self.model_used
        }

        # Add classification data if available
        if self.contract_violations is not None:
            result['contract_violations'] = self.contract_violations
        if self.novel_patterns is not None:
            result['novel_patterns'] = self.novel_patterns
        if self.research_value is not None:
            result['research_value'] = self.research_value
        if self.verification_notes is not None:
            result['verification_notes'] = self.verification_notes

        return result


@dataclass
class HumanLabel:
    """Single human label for a post."""
    rater_id: str  # R1, R2, R3
    contract_type: Optional[ContractType] = None
    pipeline_stage: Optional[PipelineStage] = None
    root_cause: Optional[RootCause] = None
    effect: Optional[Effect] = None
    notes: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rater_id': self.rater_id,
            'contract_type': self.contract_type.value if self.contract_type else None,
            'pipeline_stage': self.pipeline_stage.value if self.pipeline_stage else None,
            'root_cause': self.root_cause.value if self.root_cause else None,
            'effect': self.effect.value if self.effect else None,
            'notes': self.notes,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


@dataclass
class LabelledPost:
    """Post with complete labelling from all three raters."""
    filtered_post_id: str
    _id: Optional[str] = None

    # LLM screening results
    bulk_screening: Optional[LLMScreeningResult] = None
    borderline_screening: Optional[LLMScreeningResult] = None

    # Human labels
    label_r1: Optional[HumanLabel] = None
    label_r2: Optional[HumanLabel] = None
    label_r3: Optional[HumanLabel] = None

    # Final arbitrated result
    final_label: Optional[HumanLabel] = None
    majority_agreement: bool = False
    required_arbitration: bool = False

    # Provenance
    labelling_session_id: str = ""
    labelling_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            '_id': self._id,
            'filtered_post_id': self.filtered_post_id,
            'bulk_screening': self.bulk_screening.to_dict() if self.bulk_screening else None,
            'borderline_screening': self.borderline_screening.to_dict() if self.borderline_screening else None,
            'label_r1': self.label_r1.to_dict() if self.label_r1 else None,
            'label_r2': self.label_r2.to_dict() if self.label_r2 else None,
            'label_r3': self.label_r3.to_dict() if self.label_r3 else None,
            'final_label': self.final_label.to_dict() if self.final_label else None,
            'majority_agreement': self.majority_agreement,
            'required_arbitration': self.required_arbitration,
            'labelling_session_id': self.labelling_session_id,
            'labelling_timestamp': self.labelling_timestamp
        }


@dataclass
class LabellingSession:
    """Metadata for a labelling session."""
    _id: Optional[str] = None
    session_date: datetime = field(default_factory=datetime.utcnow)
    raters: List[str] = field(default_factory=list)  # [R1, R2, R3]
    posts_assigned: int = 0
    posts_completed: int = 0
    moderator: str = ""
    status: str = "active"  # active, completed, needs_review

    def to_dict(self) -> Dict[str, Any]:
        return {
            '_id': self._id,
            'session_date': self.session_date,
            'raters': self.raters,
            'posts_assigned': self.posts_assigned,
            'posts_completed': self.posts_completed,
            'moderator': self.moderator,
            'status': self.status
        }


@dataclass
class ReliabilityMetrics:
    """Fleiss Kappa and related reliability metrics."""
    _id: Optional[str] = None
    session_id: str = ""
    calculation_date: datetime = field(default_factory=datetime.utcnow)

    # Core metrics
    fleiss_kappa: float = 0.0
    kappa_std_error: float = 0.0
    kappa_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Sample details
    n_items: int = 0  # N items (clauses)
    n_raters: int = 3  # number of raters
    n_categories: int = 0  # k categories

    # Agreement details
    observed_agreement: float = 0.0  # P̄
    expected_agreement: float = 0.0  # P̄e

    # Per-category analysis
    category_agreement: Dict[str, float] = field(default_factory=dict)
    confusion_matrices: Dict[str, Any] = field(default_factory=dict)

    # Quality gates
    passes_threshold: bool = False  # κ ≥ 0.80
    needs_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            '_id': self._id,
            'session_id': self.session_id,
            'calculation_date': self.calculation_date,
            'fleiss_kappa': self.fleiss_kappa,
            'kappa_std_error': self.kappa_std_error,
            'kappa_confidence_interval': list(self.kappa_confidence_interval),
            'n_items': self.n_items,
            'n_raters': self.n_raters,
            'n_categories': self.n_categories,
            'observed_agreement': self.observed_agreement,
            'expected_agreement': self.expected_agreement,
            'category_agreement': self.category_agreement,
            'confusion_matrices': self.confusion_matrices,
            'passes_threshold': self.passes_threshold,
            'needs_review': self.needs_review
        }


@dataclass
class TaxonomyDefinition:
    """Taxonomy definition with hierarchical structure."""
    _id: Optional[str] = None
    name: str = ""
    version: str = "1.0.0"
    created_date: datetime = field(default_factory=datetime.utcnow)

    # Hierarchical definitions
    contract_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    root_causes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Relationships and dependencies
    type_stage_mapping: Dict[str, List[str]] = field(default_factory=dict)
    cause_effect_mapping: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            '_id': self._id,
            'name': self.name,
            'version': self.version,
            'created_date': self.created_date,
            'contract_types': self.contract_types,
            'pipeline_stages': self.pipeline_stages,
            'root_causes': self.root_causes,
            'effects': self.effects,
            'type_stage_mapping': self.type_stage_mapping,
            'cause_effect_mapping': self.cause_effect_mapping
        }
