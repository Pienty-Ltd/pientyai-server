from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from app.database.models.db_models import User, UserRole, Organization, user_organizations
from app.core.security import get_password_hash
from typing import Optional, Dict, Any, List, Tuple
import logging
from sqlalchemy import func

logger = logging.getLogger(__name__)

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_fp(self, fp: str) -> Optional[User]:
        """Get user by fingerprint"""
        try:
            result = await self.db.execute(select(User).filter(User.fp == fp))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by fp: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            result = await self.db.execute(select(User).filter(User.email == email))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            result = await self.db.execute(select(User).filter(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by id: {str(e)}")
            return None

    async def get_user_organizations(self, user_id: int, page: int = 1, per_page: int = 20) -> Tuple[List[Organization], int]:
        """Get organizations for a user with pagination"""
        try:
            offset = (page - 1) * per_page

            # Get total count
            count_stmt = (
                select(func.count(Organization.id))
                .join(user_organizations)
                .filter(user_organizations.c.user_id == user_id)
            )
            total_count = await self.db.execute(count_stmt)
            total = total_count.scalar()

            # Get paginated organizations
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
            logger.error(f"Error fetching user organizations: {str(e)}")
            raise

    async def insert_user(self, user_data: dict) -> Optional[User]:
        """Create a new user"""
        try:
            # Check if email already exists
            existing_user = await self.get_user_by_email(user_data["email"])
            if existing_user:
                logger.warning(f"Attempted to create user with existing email: {user_data['email']}")
                return None

            # Make sure hashed_password is provided
            if "password" in user_data:
                user_data["hashed_password"] = get_password_hash(user_data["password"])
                del user_data["password"]
            elif "hashed_password" not in user_data or user_data["hashed_password"] is None:
                logger.error("Cannot create user without password")
                return None

            user = User(**user_data)
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            logger.info(f"Successfully created user with email: {user.email}")
            return user

        except IntegrityError as e:
            logger.error(f"Database integrity error creating user: {str(e)}")
            await self.db.rollback()
            return None
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            await self.db.rollback()
            return None

    async def update_user(self, user_fp: str, updates: dict) -> Optional[User]:
        """Update user information"""
        try:
            user = await self.get_user_by_fp(user_fp)
            if user:
                # If updating password, hash it first
                if "password" in updates:
                    updates["hashed_password"] = get_password_hash(updates["password"])
                    del updates["password"]

                for key, value in updates.items():
                    setattr(user, key, value)
                await self.db.commit()
                await self.db.refresh(user)
                logger.info(f"Successfully updated user with fp: {user_fp}")
                return user
            return None
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            await self.db.rollback()
            return None

    async def create_user_instance_from_cache(self, cached_data: Dict[str, Any]) -> User:
        """Create a User instance from cached data without accessing the database"""
        try:
            user = User(
                id=cached_data['id'],
                email=cached_data['email'],
                full_name=cached_data['full_name'],
                is_active=cached_data['is_active'],
                fp=cached_data['fp'],
                role=UserRole(cached_data['role'])
            )

            logger.debug(f"Created User instance from cache for user: {user.email}")
            return user
        except Exception as e:
            logger.error(f"Error creating User instance from cache: {str(e)}")
            raise