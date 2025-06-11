"""
Streamlit Dashboard for LLM Contract Analysis

A comprehensive dashboard for visualizing and analyzing LLM API contract violations.
Suitable for deployment on GitHub Pages or Streamlit Cloud.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
from collections import Counter, defaultdict

# Import our modules (with error handling for deployment)
try:
    from pipeline.llm_screening.contract_taxonomy import LLMContractTaxonomy, ViolationSeverity
    from pipeline.storage.repositories import LabelledPostRepository, FilteredPostRepository
    from pipeline.llm_screening.contract_analysis import ContractAnalyzer
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    st.warning("Running in standalone mode - some features may be limited")

# Page configuration
st.set_page_config(
    page_title="LLM Contract Analysis Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .violation-card {
        background-color: #fff;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .novel-card {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class ContractDashboard:
    """Main dashboard application for contract analysis."""
    
    def __init__(self):
        self.taxonomy = LLMContractTaxonomy() if MODULES_AVAILABLE else None
        self.analyzer = ContractAnalyzer() if MODULES_AVAILABLE else None
        self.data = self._load_sample_data()
    
    def _load_sample_data(self) -> Dict[str, Any]:
        """Load sample data or real data from files."""
        # Try to load from recent analysis results
        data_files = [f for f in os.listdir('.') if f.startswith('contract_analysis_results_')]
        
        if data_files:
            # Load most recent file
            latest_file = sorted(data_files)[-1]
            with open(latest_file, 'r') as f:
                return json.load(f)
        
        # Generate sample data for demo
        return self._generate_sample_data()
    
    def _generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample data for demonstration."""
        return {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'summary': {
                'total_posts': 500,
                'posts_with_violations': 342,
                'total_violations': 756,
                'novel_violations': 89
            },
            'novel_contracts': {
                'count': 12,
                'categories': {
                    'NovelRateLimit': [
                        {'name': 'Dynamic_Rate_Adjustment', 'description': 'Rate limits that change based on usage patterns'},
                        {'name': 'Burst_Credit_System', 'description': 'Accumulated credits for burst requests'}
                    ],
                    'NovelFormat': [
                        {'name': 'Streaming_JSON_Contract', 'description': 'JSON formatting requirements for streaming responses'},
                        {'name': 'Tool_Schema_Evolution', 'description': 'Schema changes between function calling versions'}
                    ],
                    'NovelState': [
                        {'name': 'Context_Decay_Pattern', 'description': 'Performance degradation with long conversations'},
                        {'name': 'Memory_Leak_Contract', 'description': 'Memory usage patterns in stateful conversations'}
                    ]
                }
            },
            'posts': self._generate_sample_posts()
        }
    
    def _generate_sample_posts(self) -> List[Dict[str, Any]]:
        """Generate sample post data."""
        contract_types = ['RATE_LIMIT', 'CONTEXT_LENGTH', 'OUTPUT_FORMAT', 'CONTENT_POLICY', 
                         'TEMPERATURE', 'MAX_TOKENS', 'JSON_SCHEMA', 'API_KEY_FORMAT']
        severities = ['critical', 'high', 'medium', 'low']
        
        posts = []
        for i in range(50):
            num_violations = (i % 3) + 1
            violations = []
            
            for j in range(num_violations):
                is_novel = (i + j) % 7 == 0
                violation = {
                    'is_novel': is_novel,
                    'contract_type': contract_types[j % len(contract_types)] if not is_novel else None,
                    'category': 'LLM_SPECIFIC' if not is_novel else 'Novel',
                    'confidence': 0.6 + (j * 0.1),
                    'severity': severities[j % len(severities)]
                }
                
                if is_novel:
                    violation['novel_name'] = f'Novel_Pattern_{i}_{j}'
                    violation['novel_description'] = f'Discovered pattern involving {contract_types[j % len(contract_types)]}'
                
                violations.append(violation)
            
            posts.append({
                'post_id': f'post_{i}',
                'total_violations': num_violations,
                'novel_violations': sum(1 for v in violations if v['is_novel']),
                'pattern': 'single' if num_violations == 1 else 'multiple',
                'research_value': 0.4 + (i % 6) * 0.1,
                'violations': violations
            })
        
        return posts
    
    def run(self):
        """Run the main dashboard application."""
        st.title("🔍 LLM API Contract Analysis Dashboard")
        st.markdown("**Analyzing contract violations and discovering novel patterns in LLM API usage**")
        
        # Sidebar
        with st.sidebar:
            st.header("📊 Dashboard Controls")
            
            # View selector
            view = st.selectbox(
                "Select View",
                ["Overview", "Contract Violations", "Novel Discoveries", "Research Insights", "Raw Data"]
            )
            
            # Filters
            st.subheader("🔧 Filters")
            
            severity_filter = st.multiselect(
                "Severity",
                ["critical", "high", "medium", "low"],
                default=["critical", "high", "medium", "low"]
            )
            
            min_confidence = st.slider(
                "Minimum Confidence",
                0.0, 1.0, 0.3, 0.05
            )
            
            show_novel_only = st.checkbox("Show Novel Violations Only", False)
            
            # Export options
            st.subheader("📥 Export")
            if st.button("Export Report"):
                self._export_report()
        
        # Main content
        if view == "Overview":
            self._show_overview()
        elif view == "Contract Violations":
            self._show_violations(severity_filter, min_confidence, show_novel_only)
        elif view == "Novel Discoveries":
            self._show_novel_discoveries()
        elif view == "Research Insights":
            self._show_research_insights()
        elif view == "Raw Data":
            self._show_raw_data()
    
    def _show_overview(self):
        """Show overview dashboard."""
        st.header("📈 Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Posts Analyzed",
                self.data['summary']['total_posts'],
                f"{self.data['summary']['posts_with_violations']/self.data['summary']['total_posts']*100:.1f}% with violations"
            )
        
        with col2:
            st.metric(
                "Total Violations",
                self.data['summary']['total_violations'],
                f"{self.data['summary']['total_violations']/self.data['summary']['posts_with_violations']:.1f} per post"
            )
        
        with col3:
            st.metric(
                "Novel Violations",
                self.data['summary']['novel_violations'],
                f"{self.data['summary']['novel_violations']/self.data['summary']['total_violations']*100:.1f}% of total"
            )
        
        with col4:
            st.metric(
                "Novel Categories",
                len(self.data['novel_contracts']['categories']),
                f"{self.data['novel_contracts']['count']} patterns"
            )
        
        # Visualizations
        st.subheader("📊 Contract Violation Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Contract type distribution
            contract_types = []
            for post in self.data['posts']:
                for violation in post['violations']:
                    if not violation['is_novel'] and violation.get('contract_type'):
                        contract_types.append(violation['contract_type'])
            
            if contract_types:
                type_counts = Counter(contract_types)
                fig = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Contract Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution
            severities = []
            for post in self.data['posts']:
                for violation in post['violations']:
                    severities.append(violation['severity'])
            
            if severities:
                severity_counts = Counter(severities)
                severity_order = ['critical', 'high', 'medium', 'low']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[s for s in severity_order if s in severity_counts],
                        y=[severity_counts.get(s, 0) for s in severity_order if s in severity_counts],
                        marker_color=['#d32f2f', '#f57c00', '#fbc02d', '#388e3c']
                    )
                ])
                fig.update_layout(title="Violation Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Research value distribution
        st.subheader("🎯 Research Value Analysis")
        
        research_values = [post['research_value'] for post in self.data['posts']]
        
        fig = px.histogram(
            x=research_values,
            nbins=20,
            title="Research Value Distribution",
            labels={'x': 'Research Value Score', 'y': 'Number of Posts'}
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="red", 
                     annotation_text="High Value Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_violations(self, severity_filter: List[str], min_confidence: float, show_novel_only: bool):
        """Show detailed violation analysis."""
        st.header("🚨 Contract Violations Analysis")
        
        # Filter violations
        filtered_violations = []
        for post in self.data['posts']:
            for violation in post['violations']:
                if (violation['severity'] in severity_filter and 
                    violation['confidence'] >= min_confidence and
                    (not show_novel_only or violation['is_novel'])):
                    filtered_violations.append({
                        'post_id': post['post_id'],
                        **violation
                    })
        
        st.info(f"Showing {len(filtered_violations)} violations matching filters")
        
        # Violation cards
        for i, violation in enumerate(filtered_violations[:20]):  # Show top 20
            if violation['is_novel']:
                st.markdown(f"""
                <div class="novel-card">
                    <h4>🆕 Novel Pattern: {violation.get('novel_name', 'Unknown')}</h4>
                    <p><strong>Description:</strong> {violation.get('novel_description', 'No description')}</p>
                    <p><strong>Confidence:</strong> {violation['confidence']:.2f} | 
                       <strong>Severity:</strong> {violation['severity']}</p>
                    <p><small>Post: {violation['post_id']}</small></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="violation-card">
                    <h4>⚠️ {violation.get('contract_type', 'Unknown')}</h4>
                    <p><strong>Category:</strong> {violation.get('category', 'Unknown')} | 
                       <strong>Confidence:</strong> {violation['confidence']:.2f} | 
                       <strong>Severity:</strong> {violation['severity']}</p>
                    <p><small>Post: {violation['post_id']}</small></p>
                </div>
                """, unsafe_allow_html=True)
    
    def _show_novel_discoveries(self):
        """Show novel contract discoveries."""
        st.header("🌟 Novel Contract Discoveries")
        
        st.markdown("""
        These are potential new contract types discovered during analysis that don't fit 
        existing taxonomies. They represent emerging patterns in LLM API usage.
        """)
        
        # Novel categories overview
        categories = self.data['novel_contracts']['categories']
        
        # Create tabs for each category
        if categories:
            tabs = st.tabs(list(categories.keys()))
            
            for tab, (category, patterns) in zip(tabs, categories.items()):
                with tab:
                    st.subheader(f"📁 {category}")
                    
                    for pattern in patterns:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{pattern['name']}</h4>
                            <p>{pattern.get('description', 'No description available')}</p>
                            <p><strong>Severity:</strong> {pattern.get('severity', 'unknown')} | 
                               <strong>Evidence Count:</strong> {pattern.get('evidence_count', 1)}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Novel pattern timeline
        st.subheader("📅 Discovery Timeline")
        
        # Generate sample timeline data
        timeline_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            count = (i % 5) + 1 if i % 3 == 0 else 0
            timeline_data.append({
                'date': date,
                'discoveries': count
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        fig = px.line(
            df_timeline,
            x='date',
            y='discoveries',
            title="Novel Pattern Discoveries Over Time",
            markers=True
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_research_insights(self):
        """Show research insights and recommendations."""
        st.header("🔬 Research Insights")
        
        # High-value posts
        high_value_posts = [p for p in self.data['posts'] if p['research_value'] > 0.7]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📌 High Research Value Posts")
            st.info(f"Found {len(high_value_posts)} posts with research value > 0.7")
            
            # Sample high-value posts
            for post in high_value_posts[:5]:
                violation_summary = []
                for v in post['violations']:
                    if v['is_novel']:
                        violation_summary.append(f"🆕 {v.get('novel_name', 'Novel')}")
                    else:
                        violation_summary.append(f"⚠️ {v.get('contract_type', 'Unknown')}")
                
                st.markdown(f"""
                **Post {post['post_id']}** (Research Value: {post['research_value']:.2f})
                - Violations: {', '.join(violation_summary)}
                - Pattern: {post['pattern']}
                """)
        
        with col2:
            st.subheader("📊 Insights Summary")
            
            # Calculate insights
            total_posts = len(self.data['posts'])
            avg_violations = sum(p['total_violations'] for p in self.data['posts']) / total_posts
            novel_rate = sum(p['novel_violations'] for p in self.data['posts']) / sum(p['total_violations'] for p in self.data['posts'])
            
            st.metric("Avg Violations/Post", f"{avg_violations:.1f}")
            st.metric("Novel Discovery Rate", f"{novel_rate*100:.1f}%")
            st.metric("High Value Posts", f"{len(high_value_posts)}/{total_posts}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        recommendations = [
            {
                "title": "Focus on Rate Limiting Patterns",
                "description": "High frequency of rate limit violations suggests need for better documentation and tooling",
                "priority": "High"
            },
            {
                "title": "Investigate Novel Format Contracts",
                "description": "New JSON formatting requirements emerging with function calling features",
                "priority": "Medium"
            },
            {
                "title": "Document State Management Contracts",
                "description": "Context decay and memory patterns need formal specification",
                "priority": "High"
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{rec['title']}</h4>
                <p>{rec['description']}</p>
                <p><strong>Priority:</strong> {rec['priority']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _show_raw_data(self):
        """Show raw data view."""
        st.header("📄 Raw Data")
        
        # Convert to DataFrame for display
        posts_data = []
        for post in self.data['posts']:
            posts_data.append({
                'Post ID': post['post_id'],
                'Total Violations': post['total_violations'],
                'Novel Violations': post['novel_violations'],
                'Pattern': post['pattern'],
                'Research Value': post['research_value']
            })
        
        df = pd.DataFrame(posts_data)
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("Sort by", df.columns)
        with col2:
            ascending = st.checkbox("Ascending", False)
        with col3:
            show_top = st.number_input("Show top N", min_value=10, max_value=len(df), value=20)
        
        # Display sorted data
        df_sorted = df.sort_values(by=sort_by, ascending=ascending).head(show_top)
        st.dataframe(df_sorted, use_container_width=True)
        
        # Download options
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"contract_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _export_report(self):
        """Export comprehensive report."""
        report = f"""
# LLM Contract Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Posts Analyzed: {self.data['summary']['total_posts']}
- Posts with Violations: {self.data['summary']['posts_with_violations']}
- Total Violations: {self.data['summary']['total_violations']}
- Novel Violations: {self.data['summary']['novel_violations']}

## Novel Discoveries
Total Categories: {len(self.data['novel_contracts']['categories'])}

"""
        
        for category, patterns in self.data['novel_contracts']['categories'].items():
            report += f"\n### {category}\n"
            for pattern in patterns:
                report += f"- **{pattern['name']}**: {pattern.get('description', 'No description')}\n"
        
        st.download_button(
            label="📥 Download Full Report",
            data=report,
            file_name=f"contract_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        st.success("Report ready for download!")


def main():
    """Main entry point."""
    dashboard = ContractDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()