"""
Live Streamlit Dashboard for LLM Contract Analysis

This version connects directly to MongoDB collections to show real analysis results.
Falls back to sample data if database is not available.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

# Database and analysis imports
try:
    from pipeline.storage.repositories import (
        LabelledPostRepository, 
        FilteredPostRepository, 
        RawPostRepository
    )
    from pipeline.llm_screening.contract_analysis import ContractAnalyzer, analyze_post_batch
    from pipeline.llm_screening.contract_taxonomy import LLMContractTaxonomy
    from pipeline.foundation.config import ConfigManager
    from pipeline.infrastructure.database import DatabaseManager
    LIVE_DATA_AVAILABLE = True
except ImportError as e:
    LIVE_DATA_AVAILABLE = False
    st.error(f"Cannot connect to live data: {e}")

# Page configuration
st.set_page_config(
    page_title="LLM Contract Analysis - Live Data",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
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
    .data-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .live-status {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .offline-status {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


class LiveContractDashboard:
    """Live dashboard that connects to MongoDB collections."""
    
    def __init__(self):
        self.db_manager = None
        self.repositories = {}
        self.analyzer = None
        self.taxonomy = None
        self.is_connected = False
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize database connection and repositories."""
        try:
            if LIVE_DATA_AVAILABLE:
                # Initialize configuration
                self.config = ConfigManager()
                
                # Set database configuration from environment or defaults
                mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
                database_name = os.getenv('DATABASE_NAME', 'llm_contracts_research')
                
                # Initialize database manager
                asyncio.run(self._async_init(mongodb_uri, database_name))
                
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            self.is_connected = False
    
    async def _async_init(self, uri: str, db_name: str):
        """Async initialization of database components."""
        try:
            self.db_manager = DatabaseManager(uri, db_name)
            await self.db_manager.connect()
            
            # Initialize repositories
            self.repositories = {
                'raw_posts': RawPostRepository(),
                'filtered_posts': FilteredPostRepository(),
                'labelled_posts': LabelledPostRepository()
            }
            
            # Initialize analysis components
            self.analyzer = ContractAnalyzer()
            self.taxonomy = LLMContractTaxonomy()
            
            self.is_connected = True
            
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
            self.is_connected = False
    
    async def _get_live_statistics(self) -> Dict[str, Any]:
        """Get real-time statistics from database."""
        if not self.is_connected:
            return {}
        
        try:
            stats = {}
            
            # Raw posts count
            raw_posts = await self.repositories['raw_posts'].count()
            stats['raw_posts'] = raw_posts
            
            # Filtered posts count
            filtered_posts = await self.repositories['filtered_posts'].count()
            stats['filtered_posts'] = filtered_posts
            
            # Labelled posts count and breakdown
            labelled_posts = await self.repositories['labelled_posts'].get_all(limit=10000)
            stats['labelled_posts'] = len(labelled_posts)
            
            # Count positive classifications
            positive_posts = sum(1 for post in labelled_posts if post.final_decision is True)
            stats['positive_posts'] = positive_posts
            
            # Screening method breakdown
            screening_methods = {'bulk': 0, 'borderline': 0, 'agentic': 0}
            for post in labelled_posts:
                if post.bulk_screening:
                    screening_methods['bulk'] += 1
                if post.borderline_screening:
                    screening_methods['borderline'] += 1
                if post.agentic_screening:
                    screening_methods['agentic'] += 1
            
            stats['screening_methods'] = screening_methods
            
            # Platform breakdown (from raw posts)
            raw_posts_sample = await self.repositories['raw_posts'].get_all(limit=1000)
            platform_counts = Counter(post.platform.value for post in raw_posts_sample)
            stats['platforms'] = dict(platform_counts)
            
            return stats
            
        except Exception as e:
            st.error(f"Error getting live statistics: {e}")
            return {}
    
    async def _get_live_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get contract violations from live data."""
        if not self.is_connected:
            return []
        
        try:
            # Get recent labelled posts
            labelled_posts = await self.repositories['labelled_posts'].get_all(limit=limit)
            
            # Get corresponding filtered posts
            filtered_posts = []
            llm_results = {}
            
            for labelled_post in labelled_posts[:limit]:
                if labelled_post.final_decision is True:  # Only positive classifications
                    filtered_post = await self.repositories['filtered_posts'].get_by_id(
                        labelled_post.filtered_post_id
                    )
                    if filtered_post:
                        filtered_posts.append(filtered_post)
                        
                        # Get best screening result
                        llm_result = (
                            labelled_post.agentic_screening or
                            labelled_post.borderline_screening or
                            labelled_post.bulk_screening
                        )
                        if llm_result:
                            llm_results[filtered_post.id] = llm_result
            
            # Run contract analysis
            if filtered_posts:
                analysis_results = analyze_post_batch(filtered_posts, llm_results)
                
                # Convert to dashboard format
                violations = []
                for result in analysis_results:
                    if result.has_violations:
                        for violation in result.violations:
                            violations.append({
                                'post_id': result.post_id,
                                'is_novel': violation.is_novel,
                                'contract_type': violation.contract_type.value if violation.contract_type else None,
                                'category': violation.contract_category,
                                'confidence': violation.confidence,
                                'severity': violation.severity.value if violation.severity else 'unknown',
                                'novel_name': violation.novel_name,
                                'novel_description': violation.novel_description,
                                'evidence': violation.evidence[:2]  # First 2 evidence items
                            })
                
                return violations
            
        except Exception as e:
            st.error(f"Error getting live violations: {e}")
            
        return []
    
    async def _get_temporal_data(self, days: int = 30) -> pd.DataFrame:
        """Get temporal analysis data."""
        if not self.is_connected:
            return pd.DataFrame()
        
        try:
            # Get posts from last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # For demonstration, we'll use post creation dates
            raw_posts = await self.repositories['raw_posts'].get_all(limit=5000)
            
            # Filter by date and group by day
            recent_posts = [
                post for post in raw_posts 
                if post.post_created_at >= cutoff_date
            ]
            
            # Group by date
            daily_counts = defaultdict(int)
            for post in recent_posts:
                date_key = post.post_created_at.date()
                daily_counts[date_key] += 1
            
            # Convert to DataFrame
            data = []
            for date, count in daily_counts.items():
                data.append({
                    'date': date,
                    'posts': count,
                    'platform': 'combined'  # Could break down by platform
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            st.error(f"Error getting temporal data: {e}")
            return pd.DataFrame()
    
    def run(self):
        """Run the live dashboard application."""
        st.title("📊 LLM Contract Analysis - Live Dashboard")
        
        # Show connection status
        if self.is_connected:
            st.markdown("""
            <div class="data-status live-status">
                🟢 <strong>Live Data Connected</strong> - Showing real-time data from MongoDB
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="data-status offline-status">
                🟡 <strong>Offline Mode</strong> - Database connection unavailable, showing sample data
            </div>
            """, unsafe_allow_html=True)
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            
            view = st.selectbox(
                "Select View",
                ["Live Overview", "Real-time Violations", "Temporal Analysis", "Database Stats"]
            )
            
            # Refresh controls
            auto_refresh = st.checkbox("Auto-refresh (30s)", False)
            if st.button("🔄 Refresh Now") or auto_refresh:
                st.rerun()
            
            # Data limits
            st.subheader("📊 Data Limits")
            data_limit = st.slider("Max records to analyze", 50, 1000, 200)
            
            # Analysis options
            st.subheader("🔍 Analysis Options")
            include_novel_only = st.checkbox("Show only novel violations", False)
            min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.3)
        
        # Main content based on view
        if view == "Live Overview":
            self._show_live_overview(data_limit)
        elif view == "Real-time Violations":
            self._show_realtime_violations(data_limit, include_novel_only, min_confidence)
        elif view == "Temporal Analysis":
            self._show_temporal_analysis()
        elif view == "Database Stats":
            self._show_database_stats()
    
    def _show_live_overview(self, data_limit: int):
        """Show live overview with real database statistics."""
        st.header("📈 Live Overview")
        
        if self.is_connected:
            # Get real statistics
            stats = asyncio.run(self._get_live_statistics())
            
            if stats:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Raw Posts",
                        stats.get('raw_posts', 0),
                        f"From {len(stats.get('platforms', {}))} platforms"
                    )
                
                with col2:
                    st.metric(
                        "Filtered Posts",
                        stats.get('filtered_posts', 0),
                        f"{stats.get('filtered_posts', 0)/max(stats.get('raw_posts', 1), 1)*100:.1f}% retention"
                    )
                
                with col3:
                    st.metric(
                        "Screened Posts",
                        stats.get('labelled_posts', 0),
                        f"{stats.get('positive_posts', 0)} positive"
                    )
                
                with col4:
                    positive_rate = stats.get('positive_posts', 0) / max(stats.get('labelled_posts', 1), 1) * 100
                    st.metric(
                        "Positive Rate",
                        f"{positive_rate:.1f}%",
                        "Contract violations found"
                    )
                
                # Platform distribution
                if stats.get('platforms'):
                    st.subheader("📱 Platform Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            values=list(stats['platforms'].values()),
                            names=list(stats['platforms'].keys()),
                            title="Posts by Platform"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Screening methods
                        screening_data = stats.get('screening_methods', {})
                        if screening_data:
                            fig = px.bar(
                                x=list(screening_data.keys()),
                                y=list(screening_data.values()),
                                title="Screening Methods Used"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to retrieve live statistics")
        else:
            st.error("Database connection required for live overview")
    
    def _show_realtime_violations(self, data_limit: int, novel_only: bool, min_confidence: float):
        """Show real-time violations from database."""
        st.header("🚨 Real-time Contract Violations")
        
        if self.is_connected:
            # Get live violations
            violations = asyncio.run(self._get_live_violations(data_limit))
            
            # Apply filters
            filtered_violations = [
                v for v in violations
                if v['confidence'] >= min_confidence and (not novel_only or v['is_novel'])
            ]
            
            st.info(f"Found {len(filtered_violations)} violations in last {data_limit} posts")
            
            if filtered_violations:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    novel_count = sum(1 for v in filtered_violations if v['is_novel'])
                    st.metric("Novel Violations", novel_count, f"{novel_count/len(filtered_violations)*100:.1f}%")
                
                with col2:
                    avg_confidence = sum(v['confidence'] for v in filtered_violations) / len(filtered_violations)
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                
                with col3:
                    critical_count = sum(1 for v in filtered_violations if v['severity'] == 'critical')
                    st.metric("Critical Violations", critical_count)
                
                # Display violations
                st.subheader("📋 Recent Violations")
                
                for violation in filtered_violations[:20]:  # Show latest 20
                    if violation['is_novel']:
                        st.markdown(f"""
                        <div class="novel-card">
                            <h4>🆕 {violation.get('novel_name', 'Novel Pattern')}</h4>
                            <p><strong>Description:</strong> {violation.get('novel_description', 'No description')}</p>
                            <p><strong>Post:</strong> {violation['post_id']} | 
                               <strong>Confidence:</strong> {violation['confidence']:.2f} | 
                               <strong>Severity:</strong> {violation['severity']}</p>
                            {f"<p><strong>Evidence:</strong> {'; '.join(violation['evidence'][:1])}</p>" if violation['evidence'] else ""}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="violation-card">
                            <h4>⚠️ {violation.get('contract_type', 'Unknown Contract')}</h4>
                            <p><strong>Category:</strong> {violation.get('category', 'Unknown')} | 
                               <strong>Post:</strong> {violation['post_id']}</p>
                            <p><strong>Confidence:</strong> {violation['confidence']:.2f} | 
                               <strong>Severity:</strong> {violation['severity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No violations found matching current filters")
        else:
            st.error("Database connection required for real-time violations")
    
    def _show_temporal_analysis(self):
        """Show temporal analysis of data collection and violations."""
        st.header("📅 Temporal Analysis")
        
        if self.is_connected:
            # Get temporal data
            df = asyncio.run(self._get_temporal_data(30))
            
            if not df.empty:
                # Posts over time
                fig = px.line(
                    df,
                    x='date',
                    y='posts',
                    title="Posts Collected Over Time (Last 30 Days)",
                    markers=True
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_posts = df['posts'].sum()
                    st.metric("Total Posts (30 days)", total_posts)
                
                with col2:
                    avg_daily = df['posts'].mean()
                    st.metric("Avg Daily Posts", f"{avg_daily:.1f}")
                
                with col3:
                    max_daily = df['posts'].max()
                    st.metric("Peak Daily Posts", max_daily)
                
                # Show recent data table
                st.subheader("📊 Daily Breakdown")
                st.dataframe(df.sort_values('date', ascending=False).head(14), use_container_width=True)
            else:
                st.info("No temporal data available")
        else:
            st.error("Database connection required for temporal analysis")
    
    def _show_database_stats(self):
        """Show detailed database statistics."""
        st.header("🗄️ Database Statistics")
        
        if self.is_connected:
            # Collection stats
            stats = asyncio.run(self._get_live_statistics())
            
            # Display as metrics
            st.subheader("📊 Collection Sizes")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Raw Posts", stats.get('raw_posts', 0))
                st.metric("Filtered Posts", stats.get('filtered_posts', 0))
            
            with col2:
                st.metric("Labelled Posts", stats.get('labelled_posts', 0))
                st.metric("Positive Classifications", stats.get('positive_posts', 0))
            
            with col3:
                if stats.get('labelled_posts', 0) > 0:
                    filter_rate = stats.get('filtered_posts', 0) / stats.get('raw_posts', 1) * 100
                    positive_rate = stats.get('positive_posts', 0) / stats.get('labelled_posts', 1) * 100
                    st.metric("Filter Retention", f"{filter_rate:.1f}%")
                    st.metric("Positive Rate", f"{positive_rate:.1f}%")
            
            # Database info
            st.subheader("🔧 Database Configuration")
            st.text(f"URI: {os.getenv('MONGODB_URI', 'Not configured')}")
            st.text(f"Database: {os.getenv('DATABASE_NAME', 'llm_contracts_research')}")
            st.text(f"Connection Status: {'✅ Connected' if self.is_connected else '❌ Disconnected'}")
            
        else:
            st.error("Database connection required for statistics")


def main():
    """Main entry point for live dashboard."""
    # Add auto-refresh
    if st.sidebar.checkbox("Auto-refresh (30s)", False):
        import time
        time.sleep(30)
        st.rerun()
    
    dashboard = LiveContractDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()