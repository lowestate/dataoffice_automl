from typing import Any
from logging import Logger

from db.queries import insert_model_metadata, insert_training_x_model
from automl.funcs import model_result_to_dict


class TrainingPersistenceService:
    def __init__(self, *, logger: Logger) -> None:
        self.logger = logger

    async def save_model_result(
        self,
        *,
        result: dict[str, Any],
        place_in_batch: int,
        new_training_id: int,
    ) -> None:
        try:
            # 1. Получаем информацию о training_result из БД
            from db.queries import get_training_result
            tr = await get_training_result(new_training_id)
            if not tr:
                raise ValueError(f"Training result {new_training_id} not found in DB")

            user_id = tr["user_id"]
            files_minio_key = tr.get("files_minio_key") or tr.get("files_minio_key") or ""
            task_type = tr.get("task_type") or "classification"

            # 2. Восстанавливаем папку {user_id}/{filename}
            parts = files_minio_key.split("/")
            if len(parts) >= 2:
                folder_path = f"{parts[0]}/{parts[1]}"
            else:
                folder_path = f"{user_id}/{files_minio_key or 'dataset.csv'}"

            # 3. Обрабатываем файлы модели: декодируем и заливаем в MinIO
            import base64
            from setup.minio import upload_dataset_to_minio

            files = result.get("files") or {}
            updated_files = {}

            for key, file_info in files.items():
                if isinstance(file_info, dict) and "data" in file_info and "filename" in file_info:
                    base64_data = file_info["data"]
                    filename = file_info["filename"]

                    try:
                        decoded_bytes = base64.b64decode(base64_data)
                        # MinIO путь: {user_id}/{filename}/{task_type}/top-{place_in_batch}/{file_name}
                        model_minio_key = f"{folder_path}/{task_type}/top-{place_in_batch}/{filename}"
                        
                        await upload_dataset_to_minio(model_minio_key, decoded_bytes)

                        updated_files[key] = {
                            "minio_key": model_minio_key,
                            "filename": filename
                        }
                    except Exception as upload_err:
                        self.logger.error(f"Failed to upload model file {filename} to MinIO: {upload_err}")
                        updated_files[key] = file_info
                else:
                    updated_files[key] = file_info

            # Записываем обновленные легковесные пути вместо base64 данных в результат
            result["files"] = updated_files

            fields = model_result_to_dict(result, place_in_batch=place_in_batch)

            from datetime import datetime, timezone
            t_start = fields.get("train_start") or datetime.now(timezone.utc)
            t_finish = fields.get("train_finish") or datetime.now(timezone.utc)

            new_model_id = await insert_model_metadata(
                model_name=fields["model_name"],
                hyperparams=fields.get("hyperparams"),
                metrics=fields.get("metrics"),
                graphics=fields.get("graphics"),
                place_in_training_res_batch=fields.get("place_in_training_res_batch"),
                model_metadata=fields.get("model_metadata"),
                train_start=t_start,
                train_finish=t_finish,
            )

            new_txm_id = await insert_training_x_model(
                training_id=new_training_id,
                model_id=new_model_id,
            )

            self.logger.info(
                f"saved to db | model_id={new_model_id}, "
                f"training_id={new_training_id}, txm_id={new_txm_id}"
            )

        except Exception as e:
            self.logger.exception(f"error while saving to db: {str(e)}")