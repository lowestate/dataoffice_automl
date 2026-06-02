from typing import TYPE_CHECKING
from sqlalchemy import Integer, String, Boolean, Text, BigInteger
from sqlalchemy.orm import relationship, mapped_column
from sqlalchemy.orm import Session, Mapped
from sqlalchemy.sql import text
from typing import Optional

from db.base import Base
if TYPE_CHECKING:
    from db.training import TrainingResult

class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )

    username: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )

    password: Mapped[str] = mapped_column(Text, nullable=False)

    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True
    )

    status: Mapped[Optional[str]] = mapped_column(
        String(50),
        server_default=text("'default'")
    )

    reg_date: Mapped[int] = mapped_column(
        BigInteger,
        server_default=text("EXTRACT(EPOCH FROM now())::BIGINT"),
        nullable=False
    )

    trainings: Mapped[list["TrainingResult"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )

def insert_user(db: Session, user_instance: User) -> int:
    db.add(user_instance)
    db.flush()

    return user_instance.user_id

def update_user_status(db: Session, user_id: int):
    user = db.get(User, user_id)

    if not user:
        return None

    user.is_active = not user.is_active