from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import config
import urllib.parse

# Parse the database URL to handle SSL mode correctly
url = urllib.parse.urlparse(config.DATABASE_URL)
if 'sslmode' in dict(urllib.parse.parse_qsl(url.query)):
    # Remove sslmode from query and add it to connect_args
    query_dict = dict(urllib.parse.parse_qsl(url.query))
    sslmode = query_dict.pop('sslmode')
    new_query = urllib.parse.urlencode(query_dict)
    DATABASE_URL = f"{url.scheme}://{url.netloc}{url.path}?{new_query}" if new_query else f"{url.scheme}://{url.netloc}{url.path}"
    connect_args = {"ssl": sslmode == "require"}
else:
    DATABASE_URL = config.DATABASE_URL
    connect_args = {}

# Ensure the URL uses the asyncpg driver
DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')

engine = create_async_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)
Base = declarative_base()

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()