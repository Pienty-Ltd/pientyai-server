from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database.models.db_models import Organization, User

class OrganizationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_organization_by_fp(self, org_fp: str):
        result = await self.db.execute(
            select(Organization).filter(Organization.fp == org_fp)
        )
        return result.scalar_one_or_none()

    async def get_organizations_by_user(self, user_fp: str):
        result = await self.db.execute(
            select(Organization)
            .join(Organization.users)
            .filter(User.fp == user_fp)
        )
        return result.scalars().all()

    async def create_organization(self, org_data: dict):
        organization = Organization(**org_data)
        self.db.add(organization)
        await self.db.commit()
        await self.db.refresh(organization)
        return organization

    async def add_user_to_organization(self, user: User, organization: Organization):
        organization.users.append(user)
        await self.db.commit()
        await self.db.refresh(organization)
        return organization