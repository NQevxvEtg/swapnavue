# swapnavue/src/database.py
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://swapnavue_user:swapnavue_password@localhost:5432/swapnavue_db")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def init_db():
    logger.info("Initializing database connection...")
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            if result.scalar_one() == 1:
                logger.info("Successfully connected to PostgreSQL database.")
            else:
                logger.error("Failed to verify PostgreSQL connection.")
        
        with SessionLocal() as session:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
            logger.info("Ensured pgvector extension is enabled.")
        
        with SessionLocal() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS swapnavue_memories (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding VECTOR(768) NOT NULL
                );
            """))
            session.commit()
            logger.info("Ensured 'swapnavue_memories' table exists.")

        with SessionLocal() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL,
                    meta_error REAL,
                    focus REAL,
                    curiosity REAL
                );
            """))
            session.commit()
            logger.info("Ensured 'chat_messages' table exists.")
        
        with SessionLocal() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS cognitive_state_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    is_training BOOLEAN,
                    message TEXT,
                    focus REAL,
                    confidence REAL,
                    meta_error REAL,
                    curiosity REAL,
                    cognitive_stress REAL,
                    target_amplitude REAL,
                    current_amplitude REAL,
                    target_frequency REAL,
                    current_frequency REAL,
                    base_focus REAL,
                    base_curiosity REAL,
                    state_drift REAL,
                    -- NEW TM METRICS
                    predictive_accuracy REAL,
                    tm_sparsity REAL,
                    -- NEW COLUMN FOR LOSS METRICS
                    continuous_learning_loss REAL,
                    train_loss REAL,
                    val_loss REAL
                );
            """))
            session.commit()
            logger.info("Ensured 'cognitive_state_history' table exists with new TM columns.")


    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise