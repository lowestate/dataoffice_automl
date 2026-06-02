from sqlalchemy.orm import Session

from db.models import ModelMetadata, insert_model_metadata
from db.training_x_models import TrainingXModel, insert_train_x_model
from db.users import User, insert_user # type: ignore

def insert_training_info(
        db: Session, 
        model_metadata_instance: ModelMetadata,
        new_training_id: int
    ) -> tuple[int, int]:
    new_model_id = insert_model_metadata(
        db=db,
        model_metadata_instance=model_metadata_instance
    )

    txm_instance = TrainingXModel(
        model_id=new_model_id,
        training_id=new_training_id
    )
    new_txm_id = insert_train_x_model(
        db=db,
        txm_instance=txm_instance
    )

    return new_model_id, new_txm_id
