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

    async def get_organizations_by_user(self, user_id: int):
        """Get all organizations for a user"""
        try:
            result = await self.db.execute(
                select(Organization)
                .join(Organization.users)
                .filter(User.id == user_id)
            )
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organizations for user: {str(e)}")
            raise

    async def create_organization(self, org_data: dict, user: User = None):
        """Create organization and add a user to it"""
        try:
            # Create organization
            organization = Organization(**org_data)
            self.db.add(organization)

            # If user is provided, add to organization
            if user:
                organization.users.append(user)

            # Commit the transaction
            await self.db.commit()
            await self.db.refresh(organization)

            logger.info(f"Created organization: {organization.name}")
            return organization

        except Exception as e:
            logger.error(f"Error creating organization: {str(e)}")
            await self.db.rollback()
            raise

    async def delete_organization(self, org_id: int):
        """Delete an organization"""
        try:
            await self.db.execute(
                delete(Organization).where(Organization.id == org_id)
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error deleting organization: {str(e)}")
            await self.db.rollback()
            raise