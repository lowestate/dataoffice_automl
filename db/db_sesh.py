from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Fallback to standard environment variables if DB_* are not defined (e.g. when running locally)
db_user = os.getenv('DB_USERNAME') or os.getenv('POSTGRES_USER') or 'postgres'
db_password = os.getenv('DB_PASSWORD') or os.getenv('POSTGRES_PASSWORD') or 'postgres'
db_host = os.getenv('DB_HOST') or 'localhost'
db_port = os.getenv('DB_PORT') or '5432'
db_name = os.getenv('DB_NAME') or os.getenv('POSTGRES_DB') or 'ai_analyst'

DATABASE_URL = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()