from typing import TYPE_CHECKING
from sqlalchemy import Integer, Text
from sqlalchemy.orm import relationship, mapped_column
from sqlalchemy.orm import Session, Mapped

from db.base import Base
if TYPE_CHECKING:
    from db.training import TrainingResult
    
class Dataset(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )
    dataset_filename: Mapped[str] = mapped_column(Text, nullable=False)

    trainings: Mapped[list["TrainingResult"]] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan"
    )

def insert_dataset(db: Session, dataset_instance: Dataset) -> int:
    db.add(dataset_instance)
    db.flush()

    return dataset_instance.dataset_id