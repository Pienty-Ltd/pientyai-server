import logging
import random
import string
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, desc
from typing import Optional, List
from datetime import datetime
from app.database.models.invitation import InvitationCode
from app.database.models.db_models import User

logger = logging.getLogger(__name__)

class InvitationRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def generate_unique_code(self, length: int = 8) -> str:
        """Generate a unique random invitation code"""
        while True:
            # Generate a random string of letters and digits
            code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
            
            # Check if code already exists
            result = await self.db.execute(
                select(InvitationCode).where(InvitationCode.code == code)
            )
            existing_code = result.scalars().first()
            
            if not existing_code:
                return code

    async def create_invitation_code(self, description: Optional[str] = None) -> InvitationCode:
        """Create a new invitation code"""
        code = await self.generate_unique_code()
        invitation = InvitationCode(
            code=code,
            description=description,
            is_used=False
        )
        self.db.add(invitation)
        await self.db.commit()
        await self.db.refresh(invitation)
        return invitation

    async def get_invitation_code(self, code: str) -> Optional[InvitationCode]:
        """Get invitation code by its value"""
        result = await self.db.execute(
            select(InvitationCode).where(InvitationCode.code == code)
        )
        return result.scalars().first()

    async def mark_as_used(self, code: str, user_id: int) -> bool:
        """Mark invitation code as used by a specific user"""
        invitation = await self.get_invitation_code(code)
        if not invitation or invitation.is_used:
            return False
        
        invitation.is_used = True
        invitation.used_at = datetime.now()
        invitation.used_by_user_id = user_id
        await self.db.commit()
        return True

    async def get_all_invitation_codes(self, page: int = 1, per_page: int = 20, order_by_used: bool = False) -> List[InvitationCode]:
        """Get all invitation codes with pagination, ordered by created_at or used_at"""
        skip = (page - 1) * per_page
        
        if order_by_used:
            # Order by used_at date for used codes
            query = select(InvitationCode).order_by(
                desc(InvitationCode.used_at),
                desc(InvitationCode.created_at)
            ).offset(skip).limit(per_page)
        else:
            # Default order by created_at date
            query = select(InvitationCode).order_by(
                desc(InvitationCode.created_at)
            ).offset(skip).limit(per_page)
            
        result = await self.db.execute(query)
        return result.scalars().all()

    async def get_total_count(self) -> int:
        """Get total count of invitation codes"""
        result = await self.db.execute(select(InvitationCode))
        return len(result.scalars().all())

    async def delete_invitation_code(self, code_id: int) -> bool:
        """Delete an invitation code by ID"""
        result = await self.db.execute(
            select(InvitationCode).where(InvitationCode.id == code_id)
        )
        invitation = result.scalars().first()
        
        if invitation:
            await self.db.delete(invitation)
            await self.db.commit()
            return True
        return False