import os
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, JSON,
    create_engine, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

DB_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5433/employee_suggestion'
)

engine = create_engine(DB_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Suggestion(Base):
    __tablename__ = 'suggestions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    sn = Column(String(50))
    gid = Column(String(50), index=True)
    name = Column(String(200))
    title = Column(String(200))
    department = Column(String(200))
    tel = Column(String(50))
    labor_type = Column(String(50))
    shift = Column(String(50))
    user_area_name = Column(String(200))
    problem_area_name = Column(String(200))
    location_name = Column(String(200))
    manager_gid = Column(String(50))
    manager_name = Column(String(200))
    submission_date = Column(String(50))
    lean_flow_name = Column(String(200))
    description = Column(Text)
    suggestion = Column(Text)
    reply_opinion = Column(Text)
    reject_justification = Column(Text)
    status = Column(String(50))
    item_type = Column(String(200))
    good_request = Column(String(50))
    owner_gid = Column(String(50))
    owner_name = Column(String(200))
    owner_tel = Column(String(50))
    owner_manager_gid = Column(String(50))
    owner_manager_name = Column(String(200))
    score = Column(Float)
    is_repeat = Column(String(50))
    area_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_suggestions_dept', 'department'),
        Index('idx_suggestions_item_type', 'item_type'),
        Index('idx_suggestions_gid_dept', 'gid', 'department'),
    )


class AiFeedback(Base):
    __tablename__ = 'ai_feedback'

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(String(100), unique=True, index=True)
    status = Column(String(20), default='committed')
    retry_key = Column(String(100))
    gid = Column(String(50))
    description = Column(Text)
    suggestion = Column(Text)
    reply_opinion = Column(Text)
    predicted_fields = Column(JSON)
    retrieved_cases = Column(JSON)
    generated_ai_suggestions = Column(JSON)
    accepted_suggestion_ids = Column(JSON)
    edited_ai_suggestions = Column(JSON)
    discarded_suggestion_ids = Column(JSON)
    images = Column(JSON)
    submitted_at = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_ai_feedback_gid', 'gid'),
        Index('idx_ai_feedback_submitted', 'submitted_at'),
    )


class PredictionLog(Base):
    __tablename__ = 'prediction_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    gid = Column(String(50))
    description_length = Column(Integer)
    suggestion_length = Column(Integer)
    similarity = Column(Float)
    predicted_fields = Column(JSON)
    predicted_at = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
