import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db import Base

class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    flagged_count = Column(Integer, nullable=False)
    top_k = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="upload", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), nullable=True, index=True)
    row_index = Column(Integer, nullable=True)
    sensor_values = Column(JSON, nullable=False)
    imputed_sensors = Column(JSON, nullable=False)
    fail_probability = Column(Float, nullable=False)
    percentile_rank = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    is_flagged_top_k = Column(Boolean, default=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    upload = relationship("Upload", back_populates="predictions")
    shap_explanation = relationship("SHAPExplanation", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class SHAPExplanation(Base):
    __tablename__ = "shap_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True, index=True)
    shap_values = Column(JSON, nullable=False)
    top_contributors = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    prediction = relationship("Prediction", back_populates="shap_explanation")
