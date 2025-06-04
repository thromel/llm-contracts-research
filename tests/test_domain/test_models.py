"""Tests for enhanced data models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from pipeline.domain.models import (
    Platform, ContractType, PipelineStage, RootCause, Effect,
    RawPost, FilteredPost, LLMScreeningResult, HumanLabel, 
    LabelledPost, LabellingSession, ReliabilityMetrics, TaxonomyDefinition
)


class TestEnums:
    """Test enum definitions."""
    
    def test_platform_enum(self):
        """Test Platform enum values."""
        assert Platform.GITHUB == "github"
        assert Platform.STACKOVERFLOW == "stackoverflow"
    
    def test_contract_type_enum(self):
        """Test ContractType enum values."""
        assert ContractType.RATE_LIMIT == "rate_limit"
        assert ContractType.CONTEXT_LENGTH == "context_length"
        assert len(ContractType) >= 8
    
    def test_pipeline_stage_enum(self):
        """Test PipelineStage enum values."""
        assert PipelineStage.DATA_PREPROCESSING == "data_preprocessing"
        assert PipelineStage.PREDICTION == "prediction"
        assert len(PipelineStage) == 8
    
    def test_root_cause_enum(self):
        """Test RootCause enum values."""
        assert RootCause.RATE_LIMIT_EXCEEDED == "rate_limit_exceeded"
        assert RootCause.CONTEXT_OVERFLOW == "context_overflow"
        assert len(RootCause) >= 9
    
    def test_effect_enum(self):
        """Test Effect enum values."""
        assert Effect.CRASH == "crash"
        assert Effect.BAD_PERF == "bad_perf"
        assert len(Effect) >= 7


class TestRawPost:
    """Test RawPost model."""
    
    def test_valid_creation(self):
        """Test creating a valid RawPost."""
        post = RawPost(
            platform=Platform.GITHUB,
            source_id="123",
            url="https://github.com/owner/repo/issues/123",
            title="Rate limit error with OpenAI API",
            body_md="I'm getting rate limit errors...",
            author="testuser",
            post_created_at=datetime.utcnow()
        )
        
        assert post.platform == Platform.GITHUB
        assert post.source_id == "123"
        assert post.title == "Rate limit error with OpenAI API"
        assert post.id == "github_123"
        assert post.content_hash is not None
        assert len(post.content_hash) == 32  # MD5 hash length
    
    def test_content_hash_generation(self):
        """Test automatic content hash generation."""
        now = datetime.utcnow()
        post1 = RawPost(
            platform=Platform.GITHUB,
            source_id="123",
            url="https://github.com/test/repo/issues/123",
            title="Test title",
            body_md="Test body",
            author="user1",
            post_created_at=now
        )
        
        post2 = RawPost(
            platform=Platform.GITHUB,
            source_id="456",  # Different ID
            url="https://github.com/test/repo/issues/456",
            title="Test title",  # Same content
            body_md="Test body",
            author="user2",
            post_created_at=now
        )
        
        # Same content should generate same hash
        assert post1.computed_content_hash == post2.computed_content_hash
        assert post1.content_hash == post2.content_hash
    
    def test_invalid_url(self):
        """Test validation of invalid URL."""
        with pytest.raises(ValidationError, match="URL must start with"):
            RawPost(
                platform=Platform.GITHUB,
                source_id="123",
                url="invalid-url",
                title="Test",
                body_md="Test",
                author="user",
                post_created_at=datetime.utcnow()
            )
    
    def test_required_fields(self):
        """Test required field validation."""
        with pytest.raises(ValidationError):
            RawPost(
                platform=Platform.GITHUB,
                # Missing required fields
            )
    
    def test_display_summary(self):
        """Test display summary generation."""
        post = RawPost(
            platform=Platform.STACKOVERFLOW,
            source_id="456",
            url="https://stackoverflow.com/questions/456",
            title="How to handle OpenAI rate limits properly?",
            body_md="I need help...",
            author="questioner",
            post_created_at=datetime.utcnow()
        )
        
        summary = post.get_display_summary()
        assert "Stackoverflow 456:" in summary
        assert "How to handle OpenAI rate limits" in summary
    
    def test_collection_name(self):
        """Test MongoDB collection name."""
        assert RawPost.get_collection_name() == "raw_posts"
    
    def test_indexes(self):
        """Test index definitions."""
        indexes = RawPost.get_indexes()
        
        # Should have base indexes plus custom ones
        assert len(indexes) >= 7
        
        # Check for platform_source unique index
        platform_source_index = next(
            (idx for idx in indexes if idx["name"] == "platform_source_unique"), 
            None
        )
        assert platform_source_index is not None
        assert platform_source_index["unique"] is True


class TestFilteredPost:
    """Test FilteredPost model."""
    
    def test_valid_creation(self):
        """Test creating a valid FilteredPost."""
        filtered = FilteredPost(
            raw_post_id="github_123",
            passed_keyword_filter=True,
            filter_confidence=0.85,
            matched_keywords=["rate limit", "openai", "api"],
            relevant_snippets=["rate limit error", "API quota exceeded"],
            potential_contracts=["rate_limit"]
        )
        
        assert filtered.raw_post_id == "github_123"
        assert filtered.passed_keyword_filter is True
        assert filtered.filter_confidence == 0.85
        assert "rate limit" in filtered.matched_keywords
        assert len(filtered.relevant_snippets) == 2
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        filtered = FilteredPost(
            raw_post_id="test_123",
            filter_confidence=0.5
        )
        assert filtered.filter_confidence == 0.5
        
        # Test clamping of out-of-range values
        filtered = FilteredPost(
            raw_post_id="test_123",
            filter_confidence=1.5  # > 1.0
        )
        assert filtered.filter_confidence == 1.0
        
        filtered = FilteredPost(
            raw_post_id="test_123", 
            filter_confidence=-0.1  # < 0.0
        )
        assert filtered.filter_confidence == 0.0
    
    def test_should_screen_with_llm(self):
        """Test LLM screening decision logic."""
        # Should screen: passed filter with high confidence
        filtered1 = FilteredPost(
            raw_post_id="test_123",
            passed_keyword_filter=True,
            filter_confidence=0.8
        )
        assert filtered1.should_screen_with_llm() is True
        
        # Should not screen: failed filter
        filtered2 = FilteredPost(
            raw_post_id="test_456",
            passed_keyword_filter=False,
            filter_confidence=0.9
        )
        assert filtered2.should_screen_with_llm() is False
        
        # Should not screen: low confidence
        filtered3 = FilteredPost(
            raw_post_id="test_789",
            passed_keyword_filter=True,
            filter_confidence=0.2
        )
        assert filtered3.should_screen_with_llm() is False


class TestLLMScreeningResult:
    """Test LLMScreeningResult model."""
    
    def test_valid_creation(self):
        """Test creating a valid LLMScreeningResult."""
        result = LLMScreeningResult(
            decision="Y",
            confidence=0.9,
            rationale="Clear rate limit violation with API error messages",
            model_used="gpt-4-1106-preview",
            model_provider="openai",
            contract_violations=[{
                "type": "rate_limit",
                "severity": "high",
                "evidence": "429 error code"
            }]
        )
        
        assert result.decision == "Y"
        assert result.confidence == 0.9
        assert result.is_positive() is True
        assert result.is_high_confidence() is True
        assert len(result.contract_violations) == 1
    
    def test_decision_validation(self):
        """Test decision field validation."""
        # Valid decisions
        for decision in ["Y", "N", "Unsure"]:
            result = LLMScreeningResult(
                decision=decision,
                confidence=0.8,
                rationale="Test rationale",
                model_used="test-model",
                model_provider="test"
            )
            assert result.decision == decision
        
        # Invalid decision
        with pytest.raises(ValidationError, match="Decision must be Y, N, or Unsure"):
            LLMScreeningResult(
                decision="Maybe",
                confidence=0.8,
                rationale="Test rationale",
                model_used="test-model",
                model_provider="test"
            )
    
    def test_confidence_methods(self):
        """Test confidence-related methods."""
        result = LLMScreeningResult(
            decision="Y",
            confidence=0.85,
            rationale="High confidence result",
            model_used="test-model",
            model_provider="test"
        )
        
        assert result.is_high_confidence() is True
        assert result.is_high_confidence(threshold=0.9) is False
    
    def test_rationale_validation(self):
        """Test rationale minimum length validation."""
        with pytest.raises(ValidationError):
            LLMScreeningResult(
                decision="Y",
                confidence=0.8,
                rationale="Short",  # Too short
                model_used="test-model",
                model_provider="test"
            )


class TestHumanLabel:
    """Test HumanLabel model."""
    
    def test_valid_creation(self):
        """Test creating a valid HumanLabel."""
        label = HumanLabel(
            rater_id="R1",
            is_contract_violation=True,
            contract_type=ContractType.RATE_LIMIT,
            pipeline_stage=PipelineStage.PREDICTION,
            root_cause=RootCause.RATE_LIMIT_EXCEEDED,
            effect=Effect.BAD_PERF,
            notes="Clear rate limit violation",
            confidence=0.9,
            difficulty=2
        )
        
        assert label.rater_id == "R1"
        assert label.is_contract_violation is True
        assert label.contract_type == ContractType.RATE_LIMIT
        assert label.confidence == 0.9
        assert label.difficulty == 2
    
    def test_rater_id_normalization(self):
        """Test rater ID normalization."""
        # Should add R prefix and uppercase
        label1 = HumanLabel(rater_id="1", is_contract_violation=True)
        assert label1.rater_id == "R1"
        
        label2 = HumanLabel(rater_id="r2", is_contract_violation=False)
        assert label2.rater_id == "R2"
        
        label3 = HumanLabel(rater_id="R3", is_contract_violation=True)
        assert label3.rater_id == "R3"
    
    def test_get_label_summary(self):
        """Test label summary generation."""
        label = HumanLabel(
            rater_id="R1",
            is_contract_violation=True,
            contract_type=ContractType.CONTEXT_LENGTH,
            confidence=0.8,
            difficulty=3
        )
        
        summary = label.get_label_summary()
        
        assert summary["rater_id"] == "R1"
        assert summary["is_violation"] is True
        assert summary["contract_type"] == "context_length"
        assert summary["confidence"] == 0.8
        assert summary["difficulty"] == 3


class TestLabelledPost:
    """Test LabelledPost model."""
    
    def test_valid_creation(self):
        """Test creating a valid LabelledPost."""
        post = LabelledPost(
            filtered_post_id="filtered_123",
            human_labels=[
                HumanLabel(rater_id="R1", is_contract_violation=True),
                HumanLabel(rater_id="R2", is_contract_violation=True),
                HumanLabel(rater_id="R3", is_contract_violation=False)
            ],
            labelling_session_id="session_001"
        )
        
        assert post.filtered_post_id == "filtered_123"
        assert post.rater_count == 3
        assert post.positive_labels == 2
        assert post.labelling_session_id == "session_001"
    
    def test_calculate_agreement(self):
        """Test agreement calculation."""
        post = LabelledPost(
            filtered_post_id="test_123",
            human_labels=[
                HumanLabel(rater_id="R1", is_contract_violation=True),
                HumanLabel(rater_id="R2", is_contract_violation=True),
                HumanLabel(rater_id="R3", is_contract_violation=True)
            ]
        )
        
        agreement = post.calculate_agreement()
        
        assert agreement["total_raters"] == 3
        assert agreement["positive_labels"] == 3
        assert agreement["negative_labels"] == 0
        assert agreement["agreement_percentage"] == 1.0
        assert agreement["has_majority"] is True
        assert agreement["is_unanimous"] is True
    
    def test_get_final_label(self):
        """Test final label determination."""
        # Majority positive
        post1 = LabelledPost(
            filtered_post_id="test_123",
            human_labels=[
                HumanLabel(rater_id="R1", is_contract_violation=True),
                HumanLabel(rater_id="R2", is_contract_violation=True),
                HumanLabel(rater_id="R3", is_contract_violation=False)
            ]
        )
        assert post1.get_final_label() is True
        
        # Majority negative
        post2 = LabelledPost(
            filtered_post_id="test_456",
            human_labels=[
                HumanLabel(rater_id="R1", is_contract_violation=False),
                HumanLabel(rater_id="R2", is_contract_violation=False),
                HumanLabel(rater_id="R3", is_contract_violation=True)
            ]
        )
        assert post2.get_final_label() is False
        
        # No clear majority (tie)
        post3 = LabelledPost(
            filtered_post_id="test_789",
            human_labels=[
                HumanLabel(rater_id="R1", is_contract_violation=True),
                HumanLabel(rater_id="R2", is_contract_violation=False)
            ]
        )
        assert post3.get_final_label() is None


class TestLabellingSession:
    """Test LabellingSession model."""
    
    def test_valid_creation(self):
        """Test creating a valid LabellingSession."""
        session = LabellingSession(
            session_name="November 2024 Batch 1",
            raters=["R1", "R2", "R3"],
            moderator="lead_researcher",
            posts_assigned=100,
            target_fleiss_kappa=0.85
        )
        
        assert session.session_name == "November 2024 Batch 1"
        assert len(session.raters) == 3
        assert session.posts_assigned == 100
        assert session.target_fleiss_kappa == 0.85
        assert session.status == "active"
        assert session.completion_percentage == 0.0
    
    def test_progress_validation(self):
        """Test progress validation."""
        with pytest.raises(ValidationError, match="Completed posts cannot exceed assigned"):
            LabellingSession(
                session_name="Test Session",
                moderator="test_mod",
                posts_assigned=10,
                posts_completed=15  # More than assigned
            )
    
    def test_update_progress(self):
        """Test progress updates."""
        session = LabellingSession(
            session_name="Test Session",
            moderator="test_mod",
            posts_assigned=100
        )
        
        # Update progress
        session.update_progress(completed=50, in_progress=10)
        
        assert session.posts_completed == 50
        assert session.posts_in_progress == 10
        assert session.completion_percentage == 50.0
        assert session.status == "active"
        
        # Complete the session
        session.update_progress(completed=100)
        
        assert session.completion_percentage == 100.0
        assert session.status == "completed"
        assert session.actual_completion is not None


class TestReliabilityMetrics:
    """Test ReliabilityMetrics model."""
    
    def test_valid_creation(self):
        """Test creating valid ReliabilityMetrics."""
        metrics = ReliabilityMetrics(
            session_id="session_001",
            fleiss_kappa=0.82,
            kappa_std_error=0.05,
            kappa_confidence_interval=(0.72, 0.92),
            n_items=100,
            n_raters=3,
            n_categories=2,
            observed_agreement=0.85,
            expected_agreement=0.50,
            quality_threshold=0.80
        )
        
        assert metrics.fleiss_kappa == 0.82
        assert metrics.n_items == 100
        assert metrics.n_raters == 3
        assert metrics.passes_threshold is True  # 0.82 >= 0.80
    
    def test_interpret_kappa(self):
        """Test kappa interpretation."""
        test_cases = [
            (-0.1, "Poor"),
            (0.1, "Slight"),
            (0.3, "Fair"),
            (0.5, "Moderate"),
            (0.7, "Substantial"),
            (0.9, "Almost Perfect")
        ]
        
        for kappa_value, expected_interpretation in test_cases:
            metrics = ReliabilityMetrics(
                session_id="test",
                fleiss_kappa=kappa_value,
                n_items=100,
                n_raters=3,
                n_categories=2,
                observed_agreement=0.5,
                expected_agreement=0.5
            )
            
            interpretation = metrics.interpret_kappa()
            assert expected_interpretation in interpretation
    
    def test_get_quality_summary(self):
        """Test quality summary generation."""
        metrics = ReliabilityMetrics(
            session_id="test",
            fleiss_kappa=0.75,
            n_items=50,
            n_raters=3,
            n_categories=2,
            observed_agreement=0.80,
            expected_agreement=0.50,
            quality_threshold=0.80
        )
        
        summary = metrics.get_quality_summary()
        
        assert summary["fleiss_kappa"] == 0.75
        assert summary["passes_threshold"] is False
        assert summary["n_items"] == 50
        assert summary["interpretation"] == "Substantial"


class TestTaxonomyDefinition:
    """Test TaxonomyDefinition model."""
    
    def test_valid_creation(self):
        """Test creating a valid TaxonomyDefinition."""
        taxonomy = TaxonomyDefinition(
            name="LLM Contracts v2",
            version="2.1.0",
            created_by="research_team",
            contract_types={
                "rate_limit": {
                    "description": "API rate limiting violations",
                    "severity": "high"
                }
            },
            pipeline_stages={
                "prediction": {
                    "description": "Model inference stage",
                    "order": 7
                }
            }
        )
        
        assert taxonomy.name == "LLM Contracts v2"
        assert taxonomy.version == "2.1.0"
        assert taxonomy.is_active is True
        assert taxonomy.review_status == "draft"
        assert len(taxonomy.contract_types) == 1
    
    def test_version_validation(self):
        """Test semantic version validation."""
        # Valid versions
        valid_versions = ["1.0.0", "2.15.3", "10.0.1"]
        for version in valid_versions:
            taxonomy = TaxonomyDefinition(
                name="Test",
                version=version,
                created_by="test"
            )
            assert taxonomy.version == version
        
        # Invalid versions
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "1.0.x"]
        for version in invalid_versions:
            with pytest.raises(ValidationError, match="Version must be in format"):
                TaxonomyDefinition(
                    name="Test",
                    version=version,
                    created_by="test"
                )
    
    def test_get_category_count(self):
        """Test category counting."""
        taxonomy = TaxonomyDefinition(
            name="Test Taxonomy",
            version="1.0.0",
            created_by="test",
            contract_types={"type1": {}, "type2": {}},
            pipeline_stages={"stage1": {}},
            root_causes={"cause1": {}, "cause2": {}, "cause3": {}},
            effects={"effect1": {}}
        )
        
        counts = taxonomy.get_category_count()
        
        assert counts["contract_types"] == 2
        assert counts["pipeline_stages"] == 1
        assert counts["root_causes"] == 3
        assert counts["effects"] == 1
    
    def test_validate_taxonomy_completeness(self):
        """Test taxonomy completeness validation."""
        # Empty taxonomy
        empty_taxonomy = TaxonomyDefinition(
            name="Empty",
            version="1.0.0",
            created_by="test"
        )
        
        issues = empty_taxonomy.validate_taxonomy_completeness()
        assert len(issues) == 4  # All categories missing
        assert "No contract types defined" in issues
        
        # Taxonomy with orphaned mappings
        incomplete_taxonomy = TaxonomyDefinition(
            name="Incomplete",
            version="1.0.0",
            created_by="test",
            contract_types={"rate_limit": {}},
            pipeline_stages={"prediction": {}},
            root_causes={"api_error": {}},
            effects={"timeout": {}},
            type_stage_mapping={
                "rate_limit": ["prediction"],
                "unknown_type": ["unknown_stage"]  # Orphaned
            }
        )
        
        issues = incomplete_taxonomy.validate_taxonomy_completeness()
        assert any("unknown_type" in issue for issue in issues)
        assert any("unknown_stage" in issue for issue in issues)


if __name__ == "__main__":
    pytest.main([__file__])