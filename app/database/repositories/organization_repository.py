from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text, delete, desc, func
from sqlalchemy.orm import selectinload
from app.database.models.db_models import Organization, User, File
import logging

logger = logging.getLogger(__name__)

class OrganizationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_user_organization_access(self, user_id: int, organization_id: int) -> bool:
        """Check if user has access to organization using direct SQL query"""
        try:
            # Use text() for raw SQL to ensure proper async execution
            stmt = text("""
                SELECT 1
                FROM users u
                JOIN user_organizations uo ON u.id = uo.user_id
                JOIN organizations o ON o.id = uo.organization_id
                WHERE u.id = :user_id AND o.id = :organization_id
                LIMIT 1
            """)

            result = await self.db.execute(
                stmt,
                {"user_id": user_id, "organization_id": organization_id}
            )
            return result.scalar() is not None

        except Exception as e:
            logger.error(f"Error checking user organization access: {str(e)}")
            raise
            
    async def check_user_organization_access_by_fp(self, user_id: int, organization_fp: str) -> bool:
        """Check if user has access to organization by fp using direct SQL query"""
        try:
            # Use text() for raw SQL to ensure proper async execution
            stmt = text("""
                SELECT 1
                FROM users u
                JOIN user_organizations uo ON u.id = uo.user_id
                JOIN organizations o ON o.id = uo.organization_id
                WHERE u.id = :user_id AND o.fp = :organization_fp
                LIMIT 1
            """)

            result = await self.db.execute(
                stmt,
                {"user_id": user_id, "organization_fp": organization_fp}
            )
            return result.scalar() is not None

        except Exception as e:
            logger.error(f"Error checking user organization access by fp: {str(e)}")
            raise

    async def get_organization_by_id(self, org_id: int):
        """Get organization by ID with users preloaded"""
        try:
            stmt = (
                select(Organization)
                .options(selectinload(Organization.users))
                .filter(Organization.id == org_id)
            )
            result = await self.db.execute(stmt)
            organization = result.scalar_one_or_none()

            if organization:
                await self.db.refresh(organization)

            return organization

        except Exception as e:
            logger.error(f"Error fetching organization by id: {str(e)}")
            raise
            
    async def get_organization_by_fp(self, org_fp: str):
        """Get organization by FP (fingerprint) with users preloaded"""
        try:
            stmt = (
                select(Organization)
                .options(selectinload(Organization.users))
                .filter(Organization.fp == org_fp)
            )
            result = await self.db.execute(stmt)
            organization = result.scalar_one_or_none()

            if organization:
                await self.db.refresh(organization)

            return organization

        except Exception as e:
            logger.error(f"Error fetching organization by fp: {str(e)}")
            raise

    async def get_organization_files(self, org_id: int, limit: int = 20):
        """Get the latest files for an organization by ID"""
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
            
    async def get_organization_files_by_fp(self, org_fp: str, limit: int = 20):
        """Get the latest files for an organization by organization FP (fingerprint)"""
        try:
            # First get the organization by FP
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return []
                
            # Then get files
            stmt = (
                select(File)
                .filter(File.organization_id == organization.id)
                .order_by(desc(File.created_at))
                .limit(limit)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error fetching organization files by fp: {str(e)}")
            raise

    async def get_organizations_by_user(self, user_id: int, page: int = 1, per_page: int = 20):
        """Get paginated organizations for a user"""
        try:
            # Calculate offset
            offset = (page - 1) * per_page

            # Get total count
            count_stmt = (
                select(func.count(Organization.id))
                .join(Organization.users)
                .filter(User.id == user_id)
            )
            total_count = await self.db.execute(count_stmt)
            total = total_count.scalar()

            # Get paginated organizations
            stmt = (
                select(Organization)
                .join(Organization.users)
                .filter(User.id == user_id)
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
        """Delete an organization by ID"""
        try:
            await self.db.execute(
                delete(Organization).where(Organization.id == org_id)
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error deleting organization: {str(e)}")
            await self.db.rollback()
            raise
            
    async def delete_organization_by_fp(self, org_fp: str):
        """Delete an organization by FP (fingerprint)"""
        try:
            await self.db.execute(
                delete(Organization).where(Organization.fp == org_fp)
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error deleting organization by fp: {str(e)}")
            await self.db.rollback()
            raise

    async def update_organization(self, org_id: int, update_data: dict):
        """Update organization details by ID"""
        try:
            organization = await self.get_organization_by_id(org_id)
            if not organization:
                return None

            for key, value in update_data.items():
                if hasattr(organization, key):
                    setattr(organization, key, value)

            await self.db.commit()
            await self.db.refresh(organization)
            return organization

        except Exception as e:
            logger.error(f"Error updating organization: {str(e)}")
            await self.db.rollback()
            raise
            
    async def update_organization_by_fp(self, org_fp: str, update_data: dict):
        """Update organization details by FP (fingerprint)"""
        try:
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return None

            for key, value in update_data.items():
                if hasattr(organization, key):
                    setattr(organization, key, value)

            await self.db.commit()
            await self.db.refresh(organization)
            return organization

        except Exception as e:
            logger.error(f"Error updating organization by fp: {str(e)}")
            await self.db.rollback()
            raise

    async def add_user_to_organization(self, org_id: int, user_id: int):
        """Add a user to an organization by ID"""
        try:
            organization = await self.get_organization_by_id(org_id)
            if not organization:
                return False

            stmt = select(User).where(User.id == user_id)
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                return False

            organization.users.append(user)
            await self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding user to organization: {str(e)}")
            await self.db.rollback()
            raise
            
    async def add_user_to_organization_by_fp(self, org_fp: str, user_id: int):
        """Add a user to an organization by organization FP (fingerprint)"""
        try:
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return False

            stmt = select(User).where(User.id == user_id)
            result = await self.db.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                return False

            organization.users.append(user)
            await self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding user to organization by fp: {str(e)}")
            await self.db.rollback()
            raise

    async def remove_user_from_organization(self, org_id: int, user_id: int):
        """Remove a user from an organization by organization ID"""
        try:
            stmt = text("""
                DELETE FROM user_organizations
                WHERE user_id = :user_id AND organization_id = :org_id
            """)
            await self.db.execute(stmt, {"user_id": user_id, "org_id": org_id})
            await self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error removing user from organization: {str(e)}")
            await self.db.rollback()
            raise
            
    async def remove_user_from_organization_by_fp(self, org_fp: str, user_id: int):
        """Remove a user from an organization by organization FP (fingerprint)"""
        try:
            # First get the organization ID from the FP
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return False
                
            # Then remove the user from the organization
            stmt = text("""
                DELETE FROM user_organizations
                WHERE user_id = :user_id AND organization_id = :org_id
            """)
            await self.db.execute(stmt, {"user_id": user_id, "org_id": organization.id})
            await self.db.commit()
            return True

        except Exception as e:
            logger.error(f"Error removing user from organization by fp: {str(e)}")
            await self.db.rollback()
            raise

    async def get_organization_users(self, org_id: int):
        """Get list of users in an organization by ID"""
        try:
            stmt = (
                select(User)
                .join(User.organizations)
                .filter(Organization.id == org_id)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            logger.error(f"Error fetching organization users: {str(e)}")
            raise
            
    async def get_organization_users_by_fp(self, org_fp: str):
        """Get list of users in an organization by organization FP (fingerprint)"""
        try:
            # First get the organization by FP
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return []
            
            # Then get users
            stmt = (
                select(User)
                .join(User.organizations)
                .filter(Organization.id == organization.id)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            logger.error(f"Error fetching organization users by fp: {str(e)}")
            raise

    async def get_total_organization_files_count(self, org_id: int):
        """Get total number of files in an organization by ID"""
        try:
            stmt = (
                select(func.count(File.id))
                .filter(File.organization_id == org_id)
            )
            result = await self.db.execute(stmt)
            return result.scalar()

        except Exception as e:
            logger.error(f"Error getting organization files count: {str(e)}")
            raise
            
    async def get_total_organization_files_count_by_fp(self, org_fp: str):
        """Get total number of files in an organization by organization FP (fingerprint)"""
        try:
            # First get the organization by FP
            organization = await self.get_organization_by_fp(org_fp)
            if not organization:
                return 0
                
            # Then get file count
            stmt = (
                select(func.count(File.id))
                .filter(File.organization_id == organization.id)
            )
            result = await self.db.execute(stmt)
            return result.scalar()

        except Exception as e:
            logger.error(f"Error getting organization files count by fp: {str(e)}")
            raise