"""Initial migration

Revision ID: 20240220_initial
Create Date: 2024-02-20 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic
revision = '20240220_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create repositories table
    op.create_table(
        'repositories',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('github_repo_id', sa.Integer, unique=True, nullable=False),
        sa.Column('owner', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=False),
        sa.Column('url', sa.String(500), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    # Create issues table
    op.create_table(
        'issues',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('github_issue_id', sa.Integer, unique=True, nullable=False),
        sa.Column('repository_id', sa.Integer, sa.ForeignKey(
            'repositories.id'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('body', sa.Text),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('labels', JSONB, nullable=False, server_default='[]'),
        sa.Column('assignees', JSONB, nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('closed_at', sa.DateTime(timezone=True)),
    )

    # Create analysis table
    op.create_table(
        'analysis',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('issue_id', sa.Integer, sa.ForeignKey(
            'issues.id'), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('analysis_text', sa.Text, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )

    # Create indexes
    op.create_index('idx_repositories_full_name',
                    'repositories', ['full_name'])
    op.create_index('idx_issues_repository_id', 'issues', ['repository_id'])
    op.create_index('idx_issues_created_at', 'issues', ['created_at'])
    op.create_index('idx_analysis_issue_id', 'analysis', ['issue_id'])
    op.create_index('idx_analysis_user_id', 'analysis', ['user_id'])


def downgrade():
    op.drop_table('analysis')
    op.drop_table('issues')
    op.drop_table('repositories')
