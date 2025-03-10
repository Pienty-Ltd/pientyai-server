from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, delete
from sqlalchemy.orm import selectinload
from app.database.models.db_models import Organization, User
import logging

logger = logging.getLogger(__name__)

class OrganizationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_organization_by_id(self, org_id: int):
        """Get organization by ID with users preloaded"""
        try:
            result = await self.db.execute(
                select(Organization)
                .options(selectinload(Organization.users))
                .filter(Organization.id == org_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching organization by id: {str(e)}")
            raise

    async def create_organization(self, org_data: dict, user: User = None):
        """Create organization and add a user to it"""
        try:
            # Create organization
            organization = Organization(**org_data)
            self.db.add(organization)
            # Flush to get the ID and create the organization
            await self.db.flush()

            # If user is provided, add to organization
            if user:
                # Create the association
                await self.db.execute(
                    text(
                        "INSERT INTO user_organizations (user_id, organization_id) VALUES (:user_id, :org_id)"
                    ),
                    {"user_id": user.id, "org_id": organization.id}
                )

            # Commit the transaction
            await self.db.commit()

            # Refresh organization with users loaded
            await self.db.refresh(organization, ['users'])

            logger.info(f"Created organization: {organization.name}")
            return organization

        except Exception as e:
            logger.error(f"Error creating organization: {str(e)}")
            await self.db.rollback()
            raise