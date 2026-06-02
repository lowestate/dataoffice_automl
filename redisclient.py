import os
import logging
import redis
import pickle
import pandas as pd
from typing import Optional, List

from setup.logger import logger

DATASET_KEY_TTL = 15 * 60  # 15 минут


class LevelFilter(logging.Filter):
    """Фильтр для разрешения только определенных уровней логирования."""
    def __init__(self, allowed_levels: List[int]):
        super().__init__()
        self.allowed_levels = allowed_levels

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno in self.allowed_levels


class RedisClient:
    def __init__(
            self,
            name: str = "automl_back",
            log_levels: Optional[List[str]] = None,
            redis_host=os.getenv("REDIS_HOST", "redis"),
            redis_port=int(os.getenv("REDIS_PORT", 6379)),
            cache_ttl: int = 3600
        ):
        self.cache_ttl = cache_ttl

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0
        )
        logger.info(f"RedisClient initialized: Redis on {redis_host}:{redis_port}")

    # ------------------------------------------------------------------
    # Кэш датасета (df + filename) — используется временно до вызова /train
    # ------------------------------------------------------------------

    def save_to_cache(self, data_id: str, df: pd.DataFrame, filename: str):
        """Сериализует DF и метаданные в байты и сохраняет в Redis."""
        try:
            payload = {"df": df, "filename": filename}
            compressed_data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            self.redis_client.setex(f"preproc:{data_id}", self.cache_ttl, compressed_data)
        except Exception as e:
            logger.error(f"Ошибка сохранения DF в Redis: {e}")

    def get_from_cache(self, data_id: str) -> Optional[dict]:
        """Получает данные из Redis. Возвращает словарь с 'df' и 'filename'."""
        try:
            cached_data = self.redis_client.get(f"preproc:{data_id}")
            if cached_data is not None and isinstance(cached_data, bytes):
                return pickle.loads(cached_data)
        except Exception as e:
            logger.error(f"Ошибка чтения DF из Redis: {e}")
        return None

    # ------------------------------------------------------------------
    # Кэш «активного датасета» юзера: user_id → {minio_key, filename}
    # TTL = 15 минут. Один файл на юзера одновременно.
    # ------------------------------------------------------------------

    def _user_dataset_key(self, user_id: int) -> str:
        return f"dataset:user:{user_id}"

    def set_user_dataset(self, user_id: int, minio_key: str, filename: str) -> None:
        """Сохраняет minio_key активного датасета юзера с TTL 15 минут.
        Перезаписывает предыдущее значение (один датасет на юзера)."""
        try:
            payload = pickle.dumps({"minio_key": minio_key, "filename": filename})
            self.redis_client.setex(self._user_dataset_key(user_id), DATASET_KEY_TTL, payload)
            logger.info(f"set_user_dataset user_id={user_id} minio_key={minio_key}")
        except Exception as e:
            logger.error(f"Ошибка set_user_dataset user_id={user_id}: {e}")

    def get_user_dataset(self, user_id: int) -> Optional[dict]:
        """Возвращает {minio_key, filename} активного датасета юзера или None если TTL истёк."""
        try:
            raw = self.redis_client.get(self._user_dataset_key(user_id))
            if raw is not None and isinstance(raw, bytes):
                return pickle.loads(raw)
        except Exception as e:
            logger.error(f"Ошибка get_user_dataset user_id={user_id}: {e}")
        return None

    def delete_user_dataset(self, user_id: int) -> None:
        """Удаляет запись активного датасета юзера из кэша после начала обучения."""
        try:
            self.redis_client.delete(self._user_dataset_key(user_id))
            logger.info(f"delete_user_dataset user_id={user_id}")
        except Exception as e:
            logger.error(f"Ошибка delete_user_dataset user_id={user_id}: {e}")

    # ------------------------------------------------------------------
    # Прогресс обучения AutoML: training_id → {step, step_index}
    # TTL = 2 часа. Пишется из background_train_flow.
    # ------------------------------------------------------------------

    PROGRESS_TTL = 2 * 60 * 60  # 2 часа

    def _progress_key(self, training_id: int) -> str:
        return f"training_progress:{training_id}"

    def set_training_progress(self, training_id: int, step: str, step_index: int, model_name: Optional[str] = None) -> None:
        """Сохраняет текущий шаг обучения в Redis и публикует событие."""
        try:
            import json
            payload = json.dumps({"step": step, "step_index": step_index, "model_name": model_name})
            self.redis_client.setex(self._progress_key(training_id), self.PROGRESS_TTL, payload)
            self.redis_client.publish(f"channel:training_progress:{training_id}", payload)
        except Exception as e:
            logger.error(f"Ошибка set_training_progress training_id={training_id}: {e}")

    def get_training_progress(self, training_id: int) -> Optional[dict]:
        """Возвращает {step, step_index} текущего шага обучения или None."""
        try:
            import json
            raw = self.redis_client.get(self._progress_key(training_id))
            if raw is not None and isinstance(raw, bytes):
                return json.loads(raw)
        except Exception as e:
            logger.error(f"Ошибка get_training_progress training_id={training_id}: {e}")
        return None

    def clear_training_progress(self, training_id: int) -> None:
        """Удаляет запись прогресса после завершения обучения."""
        try:
            self.redis_client.delete(self._progress_key(training_id))
        except Exception as e:
            logger.error(f"Ошибка clear_training_progress training_id={training_id}: {e}")