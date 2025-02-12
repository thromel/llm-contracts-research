"""MongoDB repository implementation."""

import logging
from typing import List, Optional
from datetime import datetime

from mongoengine import connect, disconnect

from src.analysis.core.storage.base import ResultsStorage
from src.analysis.core.dto import (
    ContractAnalysisDTO,
    AnalysisMetadataDTO,
    AnalysisResultsDTO,
    GithubIssueDTO,
    CommentAnalysisDTO,
    ErrorPropagationDTO,
    NewContractDTO,
    PatternFrequencyDTO
)
from src.analysis.core.storage.mongodb.schemas import (
    GithubIssue,
    ContractAnalysis,
    AnalysisMetadata,
    AnalysisResults,
    Comment,
    CommentAnalysis,
    ErrorPropagation,
    NewContract,
    PatternFrequency
)

logger = logging.getLogger(__name__)


class MongoDBRepository(ResultsStorage):
    """MongoDB repository implementation."""

    def __init__(self, connection_uri: str, db_name: str):
        """Initialize MongoDB repository.

        Args:
            connection_uri: MongoDB connection URI
            db_name: Database name
        """
        self.connection_uri = connection_uri
        self.db_name = db_name
        self._connect()

    def _connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            connect(db=self.db_name, host=self.connection_uri)
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error("Failed to connect to MongoDB: {}".format(str(e)))
            raise

    def _disconnect(self) -> None:
        """Close MongoDB connection."""
        try:
            disconnect()
            logger.info("Disconnected from MongoDB")
        except Exception as e:
            logger.error(
                "Failed to disconnect from MongoDB: {}".format(str(e)))

    def save_results(self, analyzed_issues: List[ContractAnalysisDTO], metadata: AnalysisMetadataDTO) -> None:
        """Save analysis results to MongoDB.

        Args:
            analyzed_issues: List of analysis results
            metadata: Analysis metadata
        """
        try:
            # Save metadata
            metadata_doc = AnalysisMetadata(
                repository=metadata.repository,
                analysis_timestamp=metadata.analysis_timestamp,
                num_issues=metadata.num_issues,
                repository_url=metadata.repository_url,
                repository_owner=metadata.repository_owner,
                repository_name=metadata.repository_name,
                repository_description=metadata.repository_description,
                repository_stars=metadata.repository_stars,
                repository_forks=metadata.repository_forks,
                repository_language=metadata.repository_language,
                analysis_version=metadata.analysis_version,
                analysis_model=metadata.analysis_model,
                analysis_batch_id=metadata.analysis_batch_id
            ).save()

            # Save analyzed issues
            analysis_docs = []
            for issue in analyzed_issues:
                # Create GitHub issue document
                github_issue = self._create_github_issue(issue)

                # Create contract analysis document
                analysis = ContractAnalysis(
                    has_violation=issue.has_violation,
                    violation_type=issue.violation_type,
                    severity=issue.severity,
                    description=issue.description,
                    confidence=issue.confidence,
                    root_cause=issue.root_cause,
                    effects=issue.effects,
                    resolution_status=issue.resolution_status,
                    resolution_details=issue.resolution_details,
                    contract_category=issue.contract_category,
                    comment_analysis=self._create_comment_analysis(
                        issue.comment_analysis),
                    error_propagation=self._create_error_propagation(
                        issue.error_propagation),
                    suggested_new_contracts=[
                        self._create_new_contract(contract)
                        for contract in issue.suggested_new_contracts
                    ],
                    issue=github_issue,
                    issue_url=issue.issue_url,
                    issue_number=issue.issue_number,
                    issue_title=issue.issue_title,
                    repository_name=issue.repository_name,
                    repository_owner=issue.repository_owner,
                    analysis_timestamp=issue.analysis_timestamp
                ).save()
                analysis_docs.append(analysis)

            # Save final results document
            AnalysisResults(
                metadata=metadata_doc,
                analyzed_issues=analysis_docs
            ).save()

            logger.info("Saved analysis results to MongoDB")

        except Exception as e:
            logger.error(
                "Failed to save results to MongoDB: {}".format(str(e)))
            raise

    def load_results(self, batch_id: str) -> Optional[AnalysisResultsDTO]:
        """Load analysis results from MongoDB.

        Args:
            batch_id: Analysis batch ID

        Returns:
            Analysis results or None if not found
        """
        try:
            # Find results by batch ID
            metadata = AnalysisMetadata.objects(
                analysis_batch_id=batch_id).first()
            if not metadata:
                return None

            results = AnalysisResults.objects(metadata=metadata).first()
            if not results:
                return None

            # Convert to DTOs
            metadata_dto = self._to_metadata_dto(metadata)
            analyzed_issues = [
                self._to_analysis_dto(analysis)
                for analysis in results.analyzed_issues
            ]

            return AnalysisResultsDTO(
                metadata=metadata_dto,
                analyzed_issues=analyzed_issues
            )

        except Exception as e:
            logger.error(
                "Failed to load results from MongoDB: {}".format(str(e)))
            raise

    def _create_github_issue(self, analysis: ContractAnalysisDTO) -> GithubIssue:
        """Create GitHub issue document.

        Args:
            analysis: Analysis DTO containing issue data

        Returns:
            GitHub issue document
        """
        return GithubIssue(
            number=analysis.issue_number,
            title=analysis.issue_title,
            body="",  # We don't store the full body to save space
            state="unknown",  # State is not critical for analysis
            created_at=analysis.analysis_timestamp or datetime.now().isoformat(),
            url=analysis.issue_url,
            labels=[],  # Labels are not critical for analysis
            first_comments=[],  # Comments are analyzed but not stored to save space
            repository_name=analysis.repository_name,
            repository_owner=analysis.repository_owner,
            html_url=analysis.issue_url
        ).save()

    def _create_comment_analysis(self, analysis: CommentAnalysisDTO) -> CommentAnalysis:
        """Create comment analysis document.

        Args:
            analysis: Comment analysis DTO

        Returns:
            Comment analysis document
        """
        if not analysis:
            return None
        return CommentAnalysis(
            supporting_evidence=analysis.supporting_evidence,
            frequency=analysis.frequency,
            workarounds=analysis.workarounds,
            impact=analysis.impact
        )

    def _create_error_propagation(self, propagation: ErrorPropagationDTO) -> ErrorPropagation:
        """Create error propagation document.

        Args:
            propagation: Error propagation DTO

        Returns:
            Error propagation document
        """
        if not propagation:
            return None
        return ErrorPropagation(
            affected_stages=propagation.affected_stages,
            propagation_path=propagation.propagation_path
        )

    def _create_new_contract(self, contract: NewContractDTO) -> NewContract:
        """Create new contract document.

        Args:
            contract: New contract DTO

        Returns:
            New contract document
        """
        if not contract or not contract.pattern_frequency:
            return None
        return NewContract(
            name=contract.name,
            description=contract.description,
            rationale=contract.rationale,
            examples=contract.examples,
            parent_category=contract.parent_category,
            pattern_frequency=PatternFrequency(
                observed_count=contract.pattern_frequency.observed_count,
                confidence=contract.pattern_frequency.confidence,
                supporting_evidence=contract.pattern_frequency.supporting_evidence
            )
        )

    def _to_metadata_dto(self, metadata: AnalysisMetadata) -> AnalysisMetadataDTO:
        """Convert metadata document to DTO.

        Args:
            metadata: Metadata document

        Returns:
            Metadata DTO
        """
        return AnalysisMetadataDTO(
            repository=metadata.repository,
            analysis_timestamp=metadata.analysis_timestamp,
            num_issues=metadata.num_issues,
            repository_url=metadata.repository_url,
            repository_owner=metadata.repository_owner,
            repository_name=metadata.repository_name,
            repository_description=metadata.repository_description,
            repository_stars=metadata.repository_stars,
            repository_forks=metadata.repository_forks,
            repository_language=metadata.repository_language,
            analysis_version=metadata.analysis_version,
            analysis_model=metadata.analysis_model,
            analysis_batch_id=metadata.analysis_batch_id
        )

    def _to_analysis_dto(self, analysis: ContractAnalysis) -> ContractAnalysisDTO:
        """Convert analysis document to DTO.

        Args:
            analysis: Analysis document

        Returns:
            Analysis DTO
        """
        return ContractAnalysisDTO(
            has_violation=analysis.has_violation,
            violation_type=analysis.violation_type,
            severity=analysis.severity,
            description=analysis.description,
            confidence=analysis.confidence,
            root_cause=analysis.root_cause,
            effects=analysis.effects,
            resolution_status=analysis.resolution_status,
            resolution_details=analysis.resolution_details,
            contract_category=analysis.contract_category,
            comment_analysis=CommentAnalysisDTO(
                supporting_evidence=analysis.comment_analysis.supporting_evidence if analysis.comment_analysis else [],
                frequency=analysis.comment_analysis.frequency if analysis.comment_analysis else "",
                workarounds=analysis.comment_analysis.workarounds if analysis.comment_analysis else [],
                impact=analysis.comment_analysis.impact if analysis.comment_analysis else ""
            ),
            error_propagation=ErrorPropagationDTO(
                affected_stages=analysis.error_propagation.affected_stages if analysis.error_propagation else [],
                propagation_path=analysis.error_propagation.propagation_path if analysis.error_propagation else ""
            ),
            suggested_new_contracts=[
                NewContractDTO(
                    name=contract.name,
                    description=contract.description,
                    rationale=contract.rationale,
                    examples=contract.examples,
                    parent_category=contract.parent_category,
                    pattern_frequency=PatternFrequencyDTO(
                        observed_count=contract.pattern_frequency.observed_count,
                        confidence=contract.pattern_frequency.confidence,
                        supporting_evidence=contract.pattern_frequency.supporting_evidence
                    )
                ) for contract in analysis.suggested_new_contracts if contract
            ],
            issue_url=analysis.issue_url,
            issue_number=analysis.issue_number,
            issue_title=analysis.issue_title,
            repository_name=analysis.repository_name,
            repository_owner=analysis.repository_owner,
            analysis_timestamp=analysis.analysis_timestamp
        )
