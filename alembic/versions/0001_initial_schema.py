"""initial_schema

Revision ID: 0001
Revises: 
Create Date: 2023-12-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # We use a permissive approach here because the user might already have the table 
    # created via create_all, but might be missing the column.
    
    # Check if table exists (this is a bit hacky in standard Alembic but helpful for hybrid 'create_all' setups)
    # Ideally, we should just assume Alembic owns the schema. 
    # But to fix "face_image does not exist", we try to just add the column.
    
    # Note: 'add_column' will fail if table doesn't exist.
    # But 'create_all' likely created the table.
    
    # Use raw SQL to safe-guard against "column already exists" if run multiple times manually
    # or "table does not exist" on empty DB (though upgrades usually assume tables from prev revisions)
    
    # Since this is revision 0001, we should arguably define the whole schema.
    # However, to avoid conflicts with 'create_all', we will focus on the delta.
    
    # Let's try standard op.add_column. If it fails, the user will see why.
    # But to be robust for the user's specific error:
    
    op.add_column('face_enrollments', sa.Column('face_image', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('face_enrollments', 'face_image')
