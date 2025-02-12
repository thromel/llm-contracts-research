"""MongoDB schemas for analysis data."""

from mongoengine import (
    Document,
    EmbeddedDocument,
    StringField,
    IntField,
    FloatField,
    BooleanField,
    ListField,
    DictField,
    EmbeddedDocumentField,
    DateTimeField,
    ReferenceField
)


class Comment(EmbeddedDocument):
    """Schema for GitHub issue comments."""
    body = StringField(required=True)
    created_at = StringField(required=True)
    user = StringField()
    comment_id = IntField()
    html_url = StringField()
    updated_at = StringField()
    reactions = DictField()


class GithubIssue(Document):
    """Schema for GitHub issues."""
    number = IntField(required=True)
    title = StringField(required=True)
    body = StringField(required=True)
    state = StringField(required=True)
    created_at = StringField(required=True)
    url = StringField(required=True)
    labels = ListField(StringField())
    first_comments = ListField(EmbeddedDocumentField(Comment))
    user = StringField()
    closed_at = StringField()
    resolution_time = FloatField()
    html_url = StringField()
    issue_id = IntField()
    updated_at = StringField()
    milestone = StringField()
    assignees = ListField(StringField())
    reactions = DictField()
    repository_url = StringField()
    repository_name = StringField()
    repository_owner = StringField()

    meta = {
        'collection': 'github_issues',
        'indexes': [
            'number',
            'repository_name',
            'repository_owner',
            'created_at',
            'state'
        ]
    }


class PatternFrequency(EmbeddedDocument):
    """Schema for contract pattern frequency."""
    observed_count = IntField(required=True)
    confidence = StringField(required=True)
    supporting_evidence = StringField(required=True)


class NewContract(EmbeddedDocument):
    """Schema for suggested new contract types."""
    name = StringField(required=True)
    description = StringField(required=True)
    rationale = StringField(required=True)
    examples = ListField(StringField())
    parent_category = StringField(required=True)
    pattern_frequency = EmbeddedDocumentField(PatternFrequency)


class CommentAnalysis(EmbeddedDocument):
    """Schema for comment analysis."""
    supporting_evidence = ListField(StringField())
    frequency = StringField()
    workarounds = ListField(StringField())
    impact = StringField()


class ErrorPropagation(EmbeddedDocument):
    """Schema for error propagation."""
    affected_stages = ListField(StringField())
    propagation_path = StringField()


class ContractAnalysis(Document):
    """Schema for contract analysis results."""
    has_violation = BooleanField(required=True)
    violation_type = StringField()
    severity = StringField(required=True)
    description = StringField(required=True)
    confidence = StringField(required=True)
    root_cause = StringField(required=True)
    effects = ListField(StringField())
    resolution_status = StringField(required=True)
    resolution_details = StringField(required=True)
    contract_category = StringField(required=True)
    comment_analysis = EmbeddedDocumentField(CommentAnalysis)
    error_propagation = EmbeddedDocumentField(ErrorPropagation)
    suggested_new_contracts = ListField(EmbeddedDocumentField(NewContract))

    # Metadata and relationships
    issue = ReferenceField(GithubIssue, required=True)
    issue_url = StringField()
    issue_number = IntField()
    issue_title = StringField()
    repository_name = StringField()
    repository_owner = StringField()
    analysis_timestamp = StringField()

    meta = {
        'collection': 'contract_analyses',
        'indexes': [
            'violation_type',
            'severity',
            'contract_category',
            'repository_name',
            'analysis_timestamp'
        ]
    }


class AnalysisMetadata(Document):
    """Schema for analysis metadata."""
    repository = StringField()
    analysis_timestamp = StringField(required=True)
    num_issues = IntField(required=True)
    repository_url = StringField()
    repository_owner = StringField()
    repository_name = StringField()
    repository_description = StringField()
    repository_stars = IntField()
    repository_forks = IntField()
    repository_language = StringField()
    analysis_version = StringField()
    analysis_model = StringField()
    analysis_batch_id = StringField()

    meta = {
        'collection': 'analysis_metadata',
        'indexes': [
            'repository_name',
            'analysis_timestamp',
            'analysis_batch_id'
        ]
    }


class AnalysisResults(Document):
    """Schema for complete analysis results."""
    metadata = ReferenceField(AnalysisMetadata, required=True)
    analyzed_issues = ListField(ReferenceField(ContractAnalysis))

    meta = {
        'collection': 'analysis_results',
        'indexes': [
            'metadata'
        ]
    }
