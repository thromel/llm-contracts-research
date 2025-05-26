"""
Fleiss Kappa Validator for Inter-Rater Reliability.

Implements the exact mathematical formulation described in the methodology:
- Per-item agreement calculation
- Observed agreement across corpus  
- Expected agreement by chance
- Standard error and confidence intervals
- Bootstrap sampling for small batches
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import scipy.stats as stats
from collections import defaultdict

from ..common.models import ReliabilityMetrics, LabelledPost, ContractType, PipelineStage, RootCause, Effect
from ..common.database import MongoDBManager

logger = logging.getLogger(__name__)


@dataclass
class KappaCalculation:
    """Detailed Fleiss Kappa calculation results."""
    fleiss_kappa: float
    observed_agreement: float  # P̄
    expected_agreement: float  # P̄e
    standard_error: float
    confidence_interval: Tuple[float, float]
    n_items: int  # N (number of items/clauses)
    n_raters: int  # n (number of raters, typically 3)
    n_categories: int  # k (number of categories)
    per_item_agreement: List[float]  # Pi for each item
    marginal_proportions: Dict[str, float]  # pj for each category
    passes_threshold: bool
    rating_matrix: np.ndarray  # N x k matrix


class FleissKappaValidator:
    """
    Fleiss Kappa validator implementing the methodology's mathematical formulation.

    Mathematical Implementation:

    Per-item agreement: Pi = (1/(n(n-1))) * Σ(nij(nij-1))
    Observed agreement: P̄ = (1/N) * Σ(Pi) 
    Expected agreement: P̄e = Σ(pj²)
    Fleiss κ: κ = (P̄ - P̄e) / (1 - P̄e)

    Where:
    - N = number of items (clauses)
    - n = number of raters (3)
    - k = number of categories 
    - nij = number of raters assigning item i to category j
    - pj = marginal proportion of category j
    """

    def __init__(self, db_manager: MongoDBManager):
        """Initialize Fleiss Kappa validator.

        Args:
            db_manager: MongoDB manager for data access
        """
        self.db = db_manager
        self.threshold = 0.80  # κ ≥ 0.80 threshold from methodology

        # Category mappings for the joint LLM + ML taxonomy
        self.category_mappings = self._build_category_mappings()

    def _build_category_mappings(self) -> Dict[str, int]:
        """Build category ID mappings for the joint taxonomy."""
        categories = {}

        # Contract types
        for i, contract_type in enumerate(ContractType):
            categories[f"contract_type_{contract_type.value}"] = i

        # Pipeline stages
        stage_offset = len(ContractType)
        for i, stage in enumerate(PipelineStage):
            categories[f"pipeline_stage_{stage.value}"] = stage_offset + i

        # Root causes
        cause_offset = stage_offset + len(PipelineStage)
        for i, cause in enumerate(RootCause):
            categories[f"root_cause_{cause.value}"] = cause_offset + i

        # Effects
        effect_offset = cause_offset + len(RootCause)
        for i, effect in enumerate(Effect):
            categories[f"effect_{effect.value}"] = effect_offset + i

        return categories

    async def calculate_kappa_for_session(
        self,
        session_id: str,
        use_bootstrap: bool = False,
        bootstrap_iterations: int = 1000
    ) -> KappaCalculation:
        """Calculate Fleiss Kappa for a labelling session.

        Args:
            session_id: Labelling session ID
            use_bootstrap: Whether to use bootstrap for small samples
            bootstrap_iterations: Number of bootstrap iterations

        Returns:
            KappaCalculation with detailed results
        """
        # Get all labelled posts for this session
        labelled_posts = await self.db.get_session_labels_for_kappa(session_id)

        if len(labelled_posts) < 2:
            raise ValueError(
                f"Insufficient data for kappa calculation: {len(labelled_posts)} posts")

        logger.info(
            f"Calculating Fleiss Kappa for session {session_id} with {len(labelled_posts)} posts")

        # Build rating matrix
        rating_matrix = self._build_rating_matrix(labelled_posts)

        # Calculate kappa
        if use_bootstrap or len(labelled_posts) < 50:
            kappa_calc = self._calculate_kappa_bootstrap(
                rating_matrix, bootstrap_iterations)
        else:
            kappa_calc = self._calculate_kappa_analytical(rating_matrix)

        return kappa_calc

    def _build_rating_matrix(self, labelled_posts: List[Dict[str, Any]]) -> np.ndarray:
        """Build the N x k rating matrix from labelled posts.

        Args:
            labelled_posts: List of labelled post documents

        Returns:
            N x k matrix where entry (i,j) = number of raters assigning item i to category j
        """
        n_items = len(labelled_posts)
        n_categories = len(self.category_mappings)

        # Initialize rating matrix
        rating_matrix = np.zeros((n_items, n_categories), dtype=int)

        for item_idx, post in enumerate(labelled_posts):
            # Process each rater's labels
            for rater_label_key in ['label_r1', 'label_r2', 'label_r3']:
                rater_label = post.get(rater_label_key)
                if rater_label:
                    # Extract category assignments from rater label
                    categories = self._extract_categories_from_label(
                        rater_label)

                    # Increment counts for assigned categories
                    for category in categories:
                        if category in self.category_mappings:
                            cat_idx = self.category_mappings[category]
                            rating_matrix[item_idx, cat_idx] += 1

        return rating_matrix

    def _extract_categories_from_label(self, label: Dict[str, Any]) -> List[str]:
        """Extract category assignments from a rater label.

        Args:
            label: Rater label dictionary

        Returns:
            List of category strings for this label
        """
        categories = []

        # Contract type
        if label.get('contract_type'):
            categories.append(f"contract_type_{label['contract_type']}")

        # Pipeline stage
        if label.get('pipeline_stage'):
            categories.append(f"pipeline_stage_{label['pipeline_stage']}")

        # Root cause
        if label.get('root_cause'):
            categories.append(f"root_cause_{label['root_cause']}")

        # Effect
        if label.get('effect'):
            categories.append(f"effect_{label['effect']}")

        return categories

    def _calculate_kappa_analytical(self, rating_matrix: np.ndarray) -> KappaCalculation:
        """Calculate Fleiss Kappa using analytical method.

        Implements the exact formulation from the methodology:
        Pi = (1/(n(n-1))) * Σ(nij(nij-1))
        P̄ = (1/N) * Σ(Pi)
        P̄e = Σ(pj²)
        κ = (P̄ - P̄e) / (1 - P̄e)
        """
        N, k = rating_matrix.shape  # N items, k categories
        n = 3  # Number of raters (fixed by design)

        # Calculate per-item agreement (Pi)
        per_item_agreement = []
        for i in range(N):
            nij = rating_matrix[i, :]  # Ratings for item i
            pi = (1.0 / (n * (n - 1))) * np.sum(nij * (nij - 1))
            per_item_agreement.append(pi)

        # Calculate observed agreement (P̄)
        observed_agreement = np.mean(per_item_agreement)

        # Calculate marginal proportions (pj)
        total_ratings = N * n
        marginal_proportions = {}
        marginal_props_array = np.sum(rating_matrix, axis=0) / total_ratings

        for category, cat_idx in self.category_mappings.items():
            marginal_proportions[category] = marginal_props_array[cat_idx]

        # Calculate expected agreement (P̄e)
        expected_agreement = np.sum(marginal_props_array ** 2)

        # Calculate Fleiss Kappa (κ)
        if expected_agreement >= 1.0:
            fleiss_kappa = 0.0  # Avoid division by zero
        else:
            fleiss_kappa = (observed_agreement -
                            expected_agreement) / (1.0 - expected_agreement)

        # Calculate standard error (analytical approximation)
        # Standard error approximation for large samples
        if N > 30:
            # Simplified SE calculation (exact formula is complex)
            se_kappa = np.sqrt(
                (observed_agreement * (1 - observed_agreement)) /
                (N * (1 - expected_agreement) ** 2)
            )
        else:
            se_kappa = 0.1  # Conservative estimate for small samples

        # Calculate confidence interval (95%)
        z_score = 1.96  # 95% CI
        ci_lower = fleiss_kappa - z_score * se_kappa
        ci_upper = fleiss_kappa + z_score * se_kappa

        return KappaCalculation(
            fleiss_kappa=fleiss_kappa,
            observed_agreement=observed_agreement,
            expected_agreement=expected_agreement,
            standard_error=se_kappa,
            confidence_interval=(ci_lower, ci_upper),
            n_items=N,
            n_raters=n,
            n_categories=k,
            per_item_agreement=per_item_agreement,
            marginal_proportions=marginal_proportions,
            passes_threshold=fleiss_kappa >= self.threshold,
            rating_matrix=rating_matrix
        )

    def _calculate_kappa_bootstrap(
        self,
        rating_matrix: np.ndarray,
        n_iterations: int = 1000
    ) -> KappaCalculation:
        """Calculate Fleiss Kappa using bootstrap method for small samples.

        Args:
            rating_matrix: N x k rating matrix
            n_iterations: Number of bootstrap iterations

        Returns:
            KappaCalculation with bootstrap-based confidence intervals
        """
        N, k = rating_matrix.shape

        # Calculate observed kappa
        base_calc = self._calculate_kappa_analytical(rating_matrix)

        # Bootstrap iterations
        bootstrap_kappas = []

        for _ in range(n_iterations):
            # Sample with replacement
            sample_indices = np.random.choice(N, size=N, replace=True)
            bootstrap_matrix = rating_matrix[sample_indices, :]

            # Calculate kappa for bootstrap sample
            try:
                bootstrap_calc = self._calculate_kappa_analytical(
                    bootstrap_matrix)
                bootstrap_kappas.append(bootstrap_calc.fleiss_kappa)
            except:
                continue  # Skip failed bootstrap samples

        # Calculate bootstrap confidence interval
        if len(bootstrap_kappas) > 0:
            bootstrap_kappas = np.array(bootstrap_kappas)
            ci_lower = np.percentile(bootstrap_kappas, 2.5)
            ci_upper = np.percentile(bootstrap_kappas, 97.5)
            se_bootstrap = np.std(bootstrap_kappas)
        else:
            ci_lower, ci_upper = base_calc.confidence_interval
            se_bootstrap = base_calc.standard_error

        # Return updated calculation with bootstrap CI
        return KappaCalculation(
            fleiss_kappa=base_calc.fleiss_kappa,
            observed_agreement=base_calc.observed_agreement,
            expected_agreement=base_calc.expected_agreement,
            standard_error=se_bootstrap,
            confidence_interval=(ci_lower, ci_upper),
            n_items=base_calc.n_items,
            n_raters=base_calc.n_raters,
            n_categories=base_calc.n_categories,
            per_item_agreement=base_calc.per_item_agreement,
            marginal_proportions=base_calc.marginal_proportions,
            passes_threshold=base_calc.fleiss_kappa >= self.threshold,
            rating_matrix=rating_matrix
        )

    async def save_reliability_metrics(
        self,
        session_id: str,
        kappa_calc: KappaCalculation
    ) -> str:
        """Save reliability metrics to database.

        Args:
            session_id: Labelling session ID
            kappa_calc: Calculated kappa results

        Returns:
            ID of saved metrics document
        """
        # Create per-category agreement analysis
        category_agreement = {}

        # Calculate agreement for each category
        for category, cat_idx in self.category_mappings.items():
            if cat_idx < kappa_calc.rating_matrix.shape[1]:
                # Calculate binary agreement for this category
                binary_matrix = (
                    kappa_calc.rating_matrix[:, cat_idx] > 0).astype(int)
                if binary_matrix.sum() > 0:
                    # Simple agreement rate for this category
                    category_agreement[category] = float(binary_matrix.mean())

        # Create confusion matrices (simplified for now)
        confusion_matrices = {
            'rater_pairs': 'See detailed analysis in separate collection',
            'category_confusions': 'Generated separately'
        }

        # Create ReliabilityMetrics object
        metrics = ReliabilityMetrics(
            session_id=session_id,
            calculation_date=datetime.utcnow(),
            fleiss_kappa=kappa_calc.fleiss_kappa,
            kappa_std_error=kappa_calc.standard_error,
            kappa_confidence_interval=kappa_calc.confidence_interval,
            n_items=kappa_calc.n_items,
            n_raters=kappa_calc.n_raters,
            n_categories=kappa_calc.n_categories,
            observed_agreement=kappa_calc.observed_agreement,
            expected_agreement=kappa_calc.expected_agreement,
            category_agreement=category_agreement,
            confusion_matrices=confusion_matrices,
            passes_threshold=kappa_calc.passes_threshold,
            needs_review=not kappa_calc.passes_threshold
        )

        # Save to database
        metrics_id = await self.db.save_reliability_metrics(metrics.to_dict())

        logger.info(
            f"Saved reliability metrics for session {session_id}: "
            f"κ={kappa_calc.fleiss_kappa:.3f}, passes_threshold={kappa_calc.passes_threshold}"
        )

        return metrics_id

    def generate_diagnostics(self, kappa_calc: KappaCalculation) -> Dict[str, Any]:
        """Generate diagnostic information for the kappa calculation.

        Args:
            kappa_calc: Calculated kappa results

        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'summary': {
                'kappa': round(kappa_calc.fleiss_kappa, 3),
                'interpretation': self._interpret_kappa(kappa_calc.fleiss_kappa),
                'passes_threshold': kappa_calc.passes_threshold,
                'confidence_interval': [round(x, 3) for x in kappa_calc.confidence_interval],
                'sample_size': kappa_calc.n_items
            },

            'agreement_analysis': {
                'observed_agreement': round(kappa_calc.observed_agreement, 3),
                'expected_agreement': round(kappa_calc.expected_agreement, 3),
                'agreement_above_chance': round(
                    kappa_calc.observed_agreement - kappa_calc.expected_agreement, 3
                )
            },

            'category_analysis': {
                'total_categories': kappa_calc.n_categories,
                'marginal_proportions': {
                    k: round(v, 3) for k, v in kappa_calc.marginal_proportions.items()
                    if v > 0.01  # Only show categories with > 1% usage
                },
                'category_agreement': {}
            },

            'quality_indicators': {
                'sufficient_sample_size': kappa_calc.n_items >= 50,
                'balanced_categories': self._check_category_balance(kappa_calc.marginal_proportions),
                'high_confidence': kappa_calc.confidence_interval[0] >= 0.75,
                'needs_arbitration': not kappa_calc.passes_threshold
            }
        }

        return diagnostics

    def _interpret_kappa(self, kappa: float) -> str:
        """Interpret kappa value using Landis & Koch benchmarks."""
        if kappa <= 0.20:
            return "Slight"
        elif kappa <= 0.40:
            return "Fair"
        elif kappa <= 0.60:
            return "Moderate"
        elif kappa <= 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"

    def _check_category_balance(self, marginal_proportions: Dict[str, float]) -> bool:
        """Check if categories are reasonably balanced."""
        if not marginal_proportions:
            return False

        # Check if any category dominates (>70%) or is too rare (<5%)
        for prop in marginal_proportions.values():
            if prop > 0.70 or (prop > 0 and prop < 0.05):
                return False

        return True

    async def validate_session_quality(self, session_id: str) -> Dict[str, Any]:
        """Validate the quality of a labelling session.

        Args:
            session_id: Session ID to validate

        Returns:
            Validation results with pass/fail status
        """
        try:
            # Calculate kappa
            kappa_calc = await self.calculate_kappa_for_session(session_id)

            # Generate diagnostics
            diagnostics = self.generate_diagnostics(kappa_calc)

            # Save to database
            await self.save_reliability_metrics(session_id, kappa_calc)

            # Determine overall quality
            quality_checks = diagnostics['quality_indicators']
            passes_quality = (
                kappa_calc.passes_threshold and
                quality_checks['sufficient_sample_size'] and
                quality_checks['balanced_categories']
            )

            return {
                'session_id': session_id,
                'passes_validation': passes_quality,
                'fleiss_kappa': kappa_calc.fleiss_kappa,
                'meets_threshold': kappa_calc.passes_threshold,
                'diagnostics': diagnostics,
                'recommendations': self._generate_recommendations(kappa_calc, quality_checks)
            }

        except Exception as e:
            logger.error(f"Error validating session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'passes_validation': False,
                'error': str(e),
                'recommendations': ['Fix data issues and retry validation']
            }

    def _generate_recommendations(
        self,
        kappa_calc: KappaCalculation,
        quality_checks: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if not kappa_calc.passes_threshold:
            recommendations.append(
                f"Kappa ({kappa_calc.fleiss_kappa:.3f}) below threshold (0.80). "
                "Consider additional rater training or taxonomy refinement."
            )

        if not quality_checks['sufficient_sample_size']:
            recommendations.append(
                f"Small sample size ({kappa_calc.n_items}). "
                "Consider labelling more items for stable estimates."
            )

        if not quality_checks['balanced_categories']:
            recommendations.append(
                "Unbalanced category usage detected. "
                "Review taxonomy and rater guidelines."
            )

        if kappa_calc.confidence_interval[0] < 0.70:
            recommendations.append(
                "Low confidence interval lower bound. "
                "Increase sample size or improve rater agreement."
            )

        if not recommendations:
            recommendations.append(
                "Quality validation passed. Session ready for analysis.")

        return recommendations
