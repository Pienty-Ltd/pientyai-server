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
        logger.info(f"Fetching organization by id: {org_id}")
        try:
            result = await self.db.execute(
                select(Organization)
                .options(selectinload(Organization.users))  # Eager loading for users
                .filter(Organization.id == org_id)
            )
            org = result.scalar_one_or_none()
            if org:
                logger.info(f"Found organization: id={org.id}, name={org.name}")
            else:
                logger.warning(f"Organization not found with id: {org_id}")
            return org
        except Exception as e:
            logger.error(f"Error fetching organization by id: {str(e)}", exc_info=True)
            raise

    async def get_organizations_by_user(self, user_id: int):
        logger.info(f"Fetching organizations for user_id: {user_id}")
        try:
            stmt = (
                select(Organization)
                .options(selectinload(Organization.users))
                .join(Organization.users)
                .filter(User.id == user_id)
            )
            result = await self.db.execute(stmt)
            organizations = result.scalars().all()
            logger.info(f"Found {len(organizations)} organizations for user {user_id}")
            return organizations
        except Exception as e:
            logger.error(f"Error fetching organizations for user: {str(e)}", exc_info=True)
            raise

    async def create_organization(self, org_data: dict):
        """Create a new organization"""
        try:
            organization = Organization(**org_data)
            self.db.add(organization)
            await self.db.flush()  # Get the ID without committing
            return organization
        except Exception as e:
            logger.error(f"Error creating organization: {str(e)}", exc_info=True)
            raise

    async def add_user_to_organization(self, user: User, organization: Organization):
        """Add a user to an organization"""
        try:
            # Add user directly without refreshing or fetching
            organization.users.append(user)
            await self.db.commit()
            logger.info(f"Successfully added user {user.id} to organization {organization.id}")
            return organization
        except Exception as e:
            logger.error(f"Error adding user to organization: {str(e)}", exc_info=True)
            raise

    async def delete_organization(self, org_id: int):
        """Delete an organization"""
        try:
            await self.db.execute(
                delete(Organization).where(Organization.id == org_id)
            )
            await self.db.commit()
            logger.info(f"Successfully deleted organization with id: {org_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting organization: {str(e)}", exc_info=True)
            raise