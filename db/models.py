from typing import TYPE_CHECKING
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, mapped_column
from sqlalchemy.orm import Session, Mapped
from typing import Optional
from datetime import datetime, timezone

from db.base import Base
if TYPE_CHECKING:
    from db.training_x_models import TrainingXModel

class ModelMetadata(Base):
    __tablename__ = "models_metadata"

    model_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    model_name: Mapped[str] = mapped_column(String(255), nullable=False)

    hyperparams: Mapped[Optional[dict]] = mapped_column(JSONB)

    metrics: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    files: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    graphics: Mapped[Optional[dict]] = mapped_column(JSONB)

    place_in_training_res_batch: Mapped[Optional[int]] = mapped_column(Integer)
    
    # например кол-во кластеров, трансформация таргета и подобные специализированные параметры
    model_metadata: Mapped[Optional[dict]] = mapped_column(JSONB) 

    train_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    train_finish: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    trainings: Mapped[list["TrainingXModel"]] = relationship(
        back_populates="model",
        cascade="all, delete-orphan"
    )

def insert_model_metadata(db: Session, model_metadata_instance: ModelMetadata) -> int:
    db.add(model_metadata_instance)
    db.flush()

    return model_metadata_instance.model_id