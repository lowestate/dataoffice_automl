from typing import TYPE_CHECKING
from sqlalchemy import Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship, mapped_column, Mapped, Session
from db.base import Base

from db.base import Base
if TYPE_CHECKING:
    from db.training import TrainingResult
    from db.models import ModelMetadata

class TrainingXModel(Base):
    __tablename__ = "training_x_models"

    pair_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    training_id: Mapped[int] = mapped_column(
        ForeignKey("training_result.training_id", ondelete="CASCADE"),
        nullable=False
    )

    model_id: Mapped[int] = mapped_column(
        ForeignKey("models_metadata.model_id", ondelete="CASCADE"),
        nullable=False
    )

    training: Mapped["TrainingResult"] = relationship(
        back_populates="models"
    )

    model: Mapped["ModelMetadata"] = relationship(
        back_populates="trainings"
    )

    __table_args__ = (
        UniqueConstraint("training_id", "model_id", name="uq_training_model"),
    )

def insert_train_x_model(db: Session, txm_instance: TrainingXModel) -> int:
    db.add(txm_instance)
    db.flush()

    return txm_instance.pair_id
