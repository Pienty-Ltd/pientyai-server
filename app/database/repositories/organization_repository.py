from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, delete, desc, func
from app.database.models.db_models import Organization, User, File, user_organizations, KnowledgeBase
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class OrganizationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_organization_by_id(self, org_id: int) -> Optional[Organization]:
        """Get organization by ID"""
        try:
            stmt = select(Organization).filter(Organization.id == org_id)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching organization by id: {str(e)}")
            raise

    async def get_organization_users(self, org_id: int) -> List[User]:
        """Get users belonging to an organization using direct join"""
        try:
            stmt = (
                select(User)
                .join(user_organizations)
                .filter(user_organizations.c.organization_id == org_id)
                .order_by(User.created_at.desc())
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organization users: {str(e)}")
            raise

    async def get_organization_files(self, org_id: int, limit: int = 20) -> List[File]:
        """Get files belonging to an organization"""
        try:
            stmt = (
                select(File)
                .filter(File.organization_id == org_id)
                .order_by(desc(File.created_at))
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organization files: {str(e)}")
            raise

    async def get_organization_knowledge_base(self, org_id: int, limit: int = 20) -> List[KnowledgeBase]:
        """Get knowledge base entries for an organization"""
        try:
            stmt = (
                select(KnowledgeBase)
                .filter(KnowledgeBase.organization_id == org_id)
                .order_by(desc(KnowledgeBase.created_at))
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organization knowledge base: {str(e)}")
            raise

    async def get_organizations_by_user(self, user_id: int, page: int = 1, per_page: int = 20) -> Tuple[List[Organization], int]:
        """Get paginated organizations for a user"""
        try:
            offset = (page - 1) * per_page

            # Get total count using join
            count_stmt = (
                select(func.count(Organization.id))
                .join(user_organizations)
                .filter(user_organizations.c.user_id == user_id)
            )
            total_count = await self.db.execute(count_stmt)
            total = total_count.scalar()

            # Get paginated organizations using join
            stmt = (
                select(Organization)
                .join(user_organizations)
                .filter(user_organizations.c.user_id == user_id)
                .order_by(Organization.created_at.desc())
                .offset(offset)
                .limit(per_page)
            )
            result = await self.db.execute(stmt)
            organizations = result.scalars().all()

            return organizations, total
        except Exception as e:
            logger.error(f"Error fetching organizations for user: {str(e)}")
            raise

    async def create_organization(self, org_data: dict, user_id: int = None) -> Optional[Organization]:
        """Create organization and optionally link it to a user"""
        try:
            organization = Organization(**org_data)
            self.db.add(organization)
            await self.db.flush()  # Get the organization ID

            if user_id is not None:
                # Add user-organization association using direct insert
                stmt = user_organizations.insert().values(
                    user_id=user_id,
                    organization_id=organization.id
                )
                await self.db.execute(stmt)

            await self.db.commit()
            logger.info(f"Created organization: {organization.name}")
            return organization
        except Exception as e:
            logger.error(f"Error creating organization: {str(e)}")
            await self.db.rollback()
            raise

    async def delete_organization(self, org_id: int) -> None:
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

    async def add_user_to_organization(self, user_id: int, org_id: int) -> None:
        """Add a user to an organization using direct query"""
        try:
            stmt = user_organizations.insert().values(
                user_id=user_id,
                organization_id=org_id
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error adding user to organization: {str(e)}")
            await self.db.rollback()
            raise

    async def remove_user_from_organization(self, user_id: int, org_id: int) -> None:
        """Remove a user from an organization using direct query"""
        try:
            stmt = user_organizations.delete().where(
                user_organizations.c.user_id == user_id,
                user_organizations.c.organization_id == org_id
            )
            await self.db.execute(stmt)
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error removing user from organization: {str(e)}")
            await self.db.rollback()
            raise