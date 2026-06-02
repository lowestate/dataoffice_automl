from typing import TYPE_CHECKING
from sqlalchemy import Integer, String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, mapped_column
from sqlalchemy.orm import Session, Mapped
from typing import Optional
from datetime import datetime, timezone

from db.base import Base
if TYPE_CHECKING:
    from db.users import User
    from db.training_x_models import TrainingXModel
    
class TrainingResult(Base):
    __tablename__ = "training_result"

    training_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False
    )

    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target: Mapped[Optional[str]] = mapped_column(String(255))

    overall_train_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )
    
    overall_train_finish: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    filename: Mapped[Optional[str]] = mapped_column(String(255))
    preprocess: Mapped[Optional[dict]] = mapped_column(JSONB)

    user: Mapped["User"] = relationship(
        back_populates="trainings"
    )

    models: Mapped[list["TrainingXModel"]] = relationship(
        back_populates="training",
        cascade="all, delete-orphan"
    )

def insert_training_results(db: Session, training_instance: TrainingResult) -> int:
    db.add(training_instance)
    db.flush()

    return training_instance.training_id