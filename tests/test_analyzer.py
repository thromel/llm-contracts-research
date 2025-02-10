"""Tests for the enhanced GitHub issues analyzer."""

import json
import pytest
from unittest.mock import patch, MagicMock
from src.analysis.core.analyzer import GitHubIssuesAnalyzer


@pytest.fixture
def analyzer():
    """Create an analyzer instance for testing."""
    return GitHubIssuesAnalyzer("test/repo")


def test_analyze_issue_traditional_ml_violation(analyzer):
    """Test analyzing an issue with a traditional ML API contract violation."""
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_violation": True,
                    "violation_type": "Single_API_Method.Data_Type",
                    "severity": "high",
                    "description": "Incorrect tensor type passed to model",
                    "confidence": "high",
                    "root_cause": "Input type mismatch",
                    "effects": ["Model crash", "Invalid results"],
                    "resolution_status": "Open",
                    "resolution_details": "Convert input to float32 tensor",
                    "pipeline_stage": "preprocessing",
                    "contract_category": "Traditional ML"
                })
            }
        }]
    }

    with patch('openai.ChatCompletion.create', return_value=MagicMock(**mock_response)):
        result = analyzer.analyze_issue(
            title="Model crashes with wrong input type",
            body="The model crashes when I pass a string instead of a tensor",
            comments="This is a common issue with tensor type mismatches"
        )

        assert result["has_violation"] is True
        assert result["violation_type"] == "Single_API_Method.Data_Type"
        assert result["severity"] == "high"
        assert "tensor" in result["description"].lower()


def test_analyze_issue_llm_specific_violation(analyzer):
    """Test analyzing an issue with an LLM-specific contract violation."""
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_violation": True,
                    "violation_type": "LLM_Specific.Input_Contracts",
                    "severity": "medium",
                    "description": "Prompt exceeds maximum token limit",
                    "confidence": "high",
                    "root_cause": "Token limit exceeded",
                    "effects": ["Truncated input", "Incomplete response"],
                    "resolution_status": "Open",
                    "resolution_details": "Split prompt into smaller chunks",
                    "pipeline_stage": "input",
                    "contract_category": "LLM-specific"
                })
            }
        }]
    }

    with patch('openai.ChatCompletion.create', return_value=MagicMock(**mock_response)):
        result = analyzer.analyze_issue(
            title="Token limit exceeded in prompt",
            body="The model truncates my input when the prompt is too long",
            comments="Need to handle long prompts better"
        )

        assert result["has_violation"] is True
        assert result["violation_type"] == "LLM_Specific.Input_Contracts"
        assert result["severity"] == "medium"
        assert "token" in result["description"].lower()


def test_analyze_issue_with_contract_discovery(analyzer):
    """Test analyzing an issue that suggests a new contract type."""
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_violation": True,
                    "violation_type": "LLM_Specific.Processing_Contracts",
                    "severity": "high",
                    "description": "Model hallucination in factual response",
                    "confidence": "high",
                    "root_cause": "Insufficient fact checking",
                    "effects": ["Incorrect information provided", "User confusion"],
                    "resolution_status": "Open",
                    "resolution_details": "Implement fact verification system",
                    "pipeline_stage": "inference",
                    "contract_category": "LLM-specific",
                    "suggested_new_contracts": [
                        {
                            "name": "Factual_Consistency_Contract",
                            "description": "Ensures LLM responses maintain factual accuracy",
                            "rationale": "Recurring issues with hallucination and fact verification",
                            "examples": [
                                "Model invents non-existent references",
                                "Inconsistent facts across responses"
                            ],
                            "parent_category": "LLM_Specific.Output_Contracts"
                        }
                    ]
                })
            }
        }]
    }

    with patch('openai.ChatCompletion.create', return_value=MagicMock(**mock_response)):
        result = analyzer.analyze_issue(
            title="Model making up false information",
            body="The model is generating responses with incorrect facts",
            comments="This seems to be a common issue with factual accuracy"
        )

        assert result["has_violation"] is True
        assert "suggested_new_contracts" in result
        assert len(result["suggested_new_contracts"]) == 1
        new_contract = result["suggested_new_contracts"][0]
        assert new_contract["name"] == "Factual_Consistency_Contract"
        assert "hallucination" in new_contract["rationale"].lower()


def test_analyze_issue_with_multiple_contract_suggestions(analyzer):
    """Test analyzing an issue that suggests multiple new contract types."""
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_violation": True,
                    "violation_type": "LLM_Specific.Processing_Contracts",
                    "severity": "high",
                    "description": "Multiple API contract issues in conversation",
                    "confidence": "high",
                    "root_cause": "Lack of conversation state management",
                    "effects": ["Context loss", "Inconsistent responses"],
                    "resolution_status": "Open",
                    "resolution_details": "Implement proper state management",
                    "pipeline_stage": "processing",
                    "contract_category": "LLM-specific",
                    "suggested_new_contracts": [
                        {
                            "name": "Conversation_State_Contract",
                            "description": "Manages conversation context and state",
                            "rationale": "Need for explicit state management rules",
                            "examples": ["Context loss between turns", "State inconsistency"],
                            "parent_category": "LLM_Specific.Processing_Contracts"
                        },
                        {
                            "name": "Response_Consistency_Contract",
                            "description": "Ensures consistent responses across conversation",
                            "rationale": "Issues with contradictory responses",
                            "examples": ["Contradicting previous answers", "Inconsistent persona"],
                            "parent_category": "LLM_Specific.Output_Contracts"
                        }
                    ]
                })
            }
        }]
    }

    with patch('openai.ChatCompletion.create', return_value=MagicMock(**mock_response)):
        result = analyzer.analyze_issue(
            title="Conversation consistency issues",
            body="Model loses context and gives contradictory responses",
            comments="Multiple users reporting similar problems"
        )

        assert result["has_violation"] is True
        assert "suggested_new_contracts" in result
        assert len(result["suggested_new_contracts"]) == 2
        assert result["suggested_new_contracts"][0]["name"] == "Conversation_State_Contract"
        assert result["suggested_new_contracts"][1]["name"] == "Response_Consistency_Contract"


def test_analyze_issue_no_violation(analyzer):
    """Test analyzing an issue with no contract violation."""
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_violation": False,
                    "violation_type": None,
                    "severity": "low",
                    "description": "Feature request for new functionality",
                    "confidence": "high",
                    "root_cause": "N/A",
                    "effects": [],
                    "resolution_status": "Not applicable",
                    "resolution_details": "No contract violation found",
                    "pipeline_stage": "N/A",
                    "contract_category": "None",
                    "suggested_new_contracts": []
                })
            }
        }]
    }

    with patch('openai.ChatCompletion.create', return_value=MagicMock(**mock_response)):
        result = analyzer.analyze_issue(
            title="Add new feature",
            body="Would be great to have X feature",
            comments=None
        )

        assert result["has_violation"] is False
        assert result["violation_type"] is None
        assert result["confidence"] == "high"
        assert "suggested_new_contracts" in result
        assert len(result["suggested_new_contracts"]) == 0


def test_analyze_issue_error_handling(analyzer):
    """Test error handling in issue analysis."""
    with patch('openai.ChatCompletion.create', side_effect=Exception("API Error")):
        result = analyzer.analyze_issue(
            title="Test issue",
            body="Test body",
            comments=None
        )

        assert result["has_violation"] is False
        assert result["violation_type"] == "ERROR"
        assert result["severity"] == "low"
        assert "API Error" in result["description"]


def test_save_results(analyzer, tmp_path):
    """Test saving analysis results."""
    analyzed_issues = [
        {
            "has_violation": True,
            "violation_type": "Single_API_Method.Data_Type",
            "severity": "high",
            "description": "Test violation",
            "confidence": "high",
            "root_cause": "Test cause",
            "effects": ["Test effect"],
            "resolution_status": "Open",
            "resolution_details": "Test resolution",
            "pipeline_stage": "test",
            "contract_category": "Traditional ML",
            "suggested_new_contracts": []
        }
    ]

    # Temporarily set results_dir to tmp_path
    analyzer.results_dir = tmp_path
    analyzer.save_results(analyzed_issues)

    # Check that file was created and contains correct data
    saved_files = list(tmp_path.glob("github_issues_analysis_*.json"))
    assert len(saved_files) == 1

    with open(saved_files[0], 'r') as f:
        saved_data = json.load(f)
        assert "metadata" in saved_data
        assert "analyzed_issues" in saved_data
        assert len(saved_data["analyzed_issues"]) == 1
        assert saved_data["analyzed_issues"][0]["violation_type"] == "Single_API_Method.Data_Type"
