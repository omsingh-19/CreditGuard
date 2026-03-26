from sqlalchemy.engine import create_engine
from sqlalchemy.orm import declarative_base , sessionmaker

URL = "sqlite:///./CreditGurad.db"

engine = create_engine(URL,connect_args={"check_same_thread": False})

session_local = sessionmaker(bind = engine,autoflush=False,autocommit=False)

Base = declarative_base()

def get_db():
    db = session_local()
    try: 
        yield db
    finally:
        db.close()
    