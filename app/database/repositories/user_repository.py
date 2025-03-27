from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, desc
from app.database.models.db_models import User, UserRole, File, Organization
from app.core.security import get_password_hash
from typing import Optional, Dict, Any, List, Tuple
import logging
import math

logger = logging.getLogger(__name__)

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_fp(self, fp: str) -> Optional[User]:
        try:
            result = await self.db.execute(select(User).filter(User.fp == fp))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by fp: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        try:
            result = await self.db.execute(select(User).filter(User.email == email))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None

    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        try:
            result = await self.db.execute(select(User).filter(User.id == user_id))
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting user by id: {str(e)}")
            return None

    async def insert_user(self, user_data: dict) -> Optional[User]:
        try:
            # Check if email already exists
            existing_user = await self.get_user_by_email(user_data["email"])
            if existing_user:
                logger.warning(f"Attempted to create user with existing email: {user_data['email']}")
                return None

            # Make sure hashed_password is provided and not None
            if "password" in user_data:
                user_data["hashed_password"] = get_password_hash(user_data["password"])
                del user_data["password"]
            elif "hashed_password" not in user_data or user_data["hashed_password"] is None:
                logger.error("Cannot create user without password")
                return None

            user = User(**user_data)
            self.db.add(user)
            await self.db.commit()  # This will create a new transaction
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
        """
        Redis'ten alınan kullanıcı verilerinden bir User nesnesi oluşturur.
        Bu metod veritabanına erişmez, sadece bellekte bir User nesnesi oluşturur.
        """
        try:
            # User nesnesini cached verilerle oluştur
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
            
    async def get_users_paginated(self, page: int = 1, per_page: int = 20) -> Tuple[List[User], int]:
        """
        Get all users with pagination
        
        Args:
            page: Page number (starting from 1)
            per_page: Number of items per page
            
        Returns:
            Tuple of (list of users, total count)
        """
        try:
            # Count total users
            count_stmt = select(func.count(User.id))
            result = await self.db.execute(count_stmt)
            total_count = result.scalar() or 0
            
            # Get paginated users
            offset = (page - 1) * per_page
            stmt = (
                select(User)
                .order_by(desc(User.created_at))
                .offset(offset)
                .limit(per_page)
            )
            
            result = await self.db.execute(stmt)
            users = result.scalars().all()
            
            return users, total_count
            
        except Exception as e:
            logger.error(f"Error getting paginated users: {str(e)}")
            raise
            
    async def get_user_organizations(self, user_fp: str) -> List[Organization]:
        """
        Get all organizations a user belongs to
        
        Args:
            user_fp: User fingerprint
            
        Returns:
            List of organizations
        """
        try:
            user = await self.get_user_by_fp(user_fp)
            if not user:
                logger.warning(f"Cannot get organizations for non-existent user: {user_fp}")
                return []
                
            # Organizations are already loaded with lazy="selectin"
            return user.organizations
            
        except Exception as e:
            logger.error(f"Error getting user organizations: {str(e)}")
            raise
            
    async def get_user_files_paginated(self, user_fp: str, page: int = 1, per_page: int = 20) -> Tuple[List[File], int]:
        """
        Get all files uploaded by a user with pagination
        
        Args:
            user_fp: User fingerprint
            page: Page number (starting from 1)
            per_page: Number of items per page
            
        Returns:
            Tuple of (list of files, total count)
        """
        try:
            user = await self.get_user_by_fp(user_fp)
            if not user:
                logger.warning(f"Cannot get files for non-existent user: {user_fp}")
                return [], 0
                
            # Count total files
            count_stmt = select(func.count(File.id)).where(File.user_id == user.id)
            result = await self.db.execute(count_stmt)
            total_count = result.scalar() or 0
            
            # Get paginated files
            offset = (page - 1) * per_page
            stmt = (
                select(File)
                .where(File.user_id == user.id)
                .order_by(desc(File.created_at))
                .offset(offset)
                .limit(per_page)
            )
            
            result = await self.db.execute(stmt)
            files = result.scalars().all()
            
            return files, total_count
            
        except Exception as e:
            logger.error(f"Error getting user files: {str(e)}")
            raise