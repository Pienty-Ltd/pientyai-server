from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import User

class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_by_fp(self, fp: str):
        result = await self.db.execute(select(User).filter(User.fp == fp))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str):
        result = await self.db.execute(select(User).filter(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: int):
        result = await self.db.execute(select(User).filter(User.id == user_id))
        return result.scalar_one_or_none()

    async def insert_user(self, user_model: dict):
        user = User(**user_model)
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def update_user(self, user_id: int, updates: dict):
        await self.db.execute(
            select(User).where(User.id == user_id).update(updates, synchronize_session="fetch")
        )
        await self.db.commit()
        return await self.get_user_by_id(user_id)