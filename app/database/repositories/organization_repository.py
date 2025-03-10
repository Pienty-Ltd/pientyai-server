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
        logger.info(f"Creating new organization with data: {org_data}")
        try:
            organization = Organization(**org_data)
            self.db.add(organization)
            await self.db.commit()  # Directly commit the organization
            await self.db.refresh(organization)
            logger.info(f"Created organization: id={organization.id}, name={organization.name}")
            return organization
        except Exception as e:
            logger.error(f"Error creating organization: {str(e)}", exc_info=True)
            await self.db.rollback()
            raise

    async def add_user_to_organization(self, user: User, organization: Organization):
        logger.info(f"Adding user {user.id} to organization {organization.id}")
        try:
            # Get fresh organization instance with users loaded
            org = await self.get_organization_by_id(organization.id)
            if not org:
                raise Exception(f"Organization {organization.id} not found")

            # Add user to organization
            org.users.append(user)
            await self.db.commit()
            await self.db.refresh(org)
            logger.info(f"Successfully added user {user.id} to organization {organization.id}")
            return org
        except Exception as e:
            logger.error(f"Error adding user to organization: {str(e)}", exc_info=True)
            await self.db.rollback()
            raise

    async def delete_organization(self, org_id: int):
        logger.info(f"Deleting organization with id: {org_id}")
        try:
            # Önce organizasyonu getir
            organization = await self.get_organization_by_id(org_id)
            if not organization:
                logger.warning(f"Organization not found with id: {org_id}")
                return False

            # Organizasyonu sil
            await self.db.execute(
                delete(Organization).where(Organization.id == org_id)
            )
            await self.db.commit()
            logger.info(f"Successfully deleted organization with id: {org_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting organization: {str(e)}", exc_info=True)
            await self.db.rollback()
            raise