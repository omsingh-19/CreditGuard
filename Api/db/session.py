from sqlalchemy.ext.asyncio import create_async_engine , async_sessionmaker
from sqlalchemy.orm import declarative_base 
from Api.config import settings

URL = settings.database_url

engine = create_async_engine(URL)

session_local = async_sessionmaker(bind = engine,autoflush=False,autocommit=False,expire_on_commit=False)

Base = declarative_base()

async def get_db():
    async with session_local() as session:
        yield session
    