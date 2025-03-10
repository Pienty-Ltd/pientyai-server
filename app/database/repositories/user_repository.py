from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from app.database.models.db_models import User, UserRole
from typing import Optional, Dict, Any
import logging

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

        Args:
            cached_data: Redis'ten alınan kullanıcı verileri

        Returns:
            User: Oluşturulan User nesnesi
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