"""
Все операции с БД для automl через async psycopg pool.
Никакого SQLAlchemy engine / sessionmaker / Depends.
"""
import json
from datetime import datetime, timezone
from typing import Optional, Any

from psycopg import sql
from db.pool import pool
from setup.logger import logger


async def run_startup_migrations() -> None:
    """
    Приводит схему к актуальному виду для БД созданных до текущей версии миграции.
    Безопасно добавляет недостающие колонки через IF NOT EXISTS.
    """
    async with pool.connection() as conn:
        # status в training_result — добавлялся позже, нужен для старых БД
        await conn.execute(
            "ALTER TABLE training_result ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'completed'"
        )
        # train_start / train_finish в models_metadata — добавлялись позже
        await conn.execute(
            "ALTER TABLE models_metadata ADD COLUMN IF NOT EXISTS train_start TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
        )
        await conn.execute(
            "ALTER TABLE models_metadata ADD COLUMN IF NOT EXISTS train_finish TIMESTAMP WITH TIME ZONE DEFAULT NOW()"
        )
        
        # Создаем таблицу automl_chats для обратной совместимости, если ее нет
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS automl_chats (
                chat_id VARCHAR(50) PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                dataset_name VARCHAR(255) NOT NULL,
                files_minio_key VARCHAR(200) NOT NULL,
                training_id INTEGER REFERENCES training_result(training_id) ON DELETE SET NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """
        )

        # Проверяем наличие старых колонок user_id / files_minio_key в training_result
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'training_result' AND column_name = 'user_id'
                """
            )
            row = await cur.fetchone()
            has_old_cols = row and row[0] > 0

        if has_old_cols:
            # Импортируем существующие обучения в automl_chats, чтобы сохранить историю пользователя
            await conn.execute(
                """
                INSERT INTO automl_chats (chat_id, user_id, dataset_name, files_minio_key, training_id, created_at, updated_at)
                SELECT 
                    MD5(random()::text || clock_timestamp()::text)::varchar(50) as chat_id,
                    tr.user_id,
                    CASE 
                        WHEN tr.files_minio_key LIKE '%/%' THEN split_part(tr.files_minio_key, '/', cardinality(string_to_array(tr.files_minio_key, '/')))
                        ELSE COALESCE(tr.files_minio_key, 'Dataset')
                    END as dataset_name,
                    tr.files_minio_key,
                    tr.training_id,
                    tr.overall_train_start,
                    tr.overall_train_finish
                FROM training_result tr
                WHERE NOT EXISTS (
                    SELECT 1 FROM automl_chats ac WHERE ac.training_id = tr.training_id
                )
                """
            )
            # Удаляем дублирующиеся колонки из training_result
            await conn.execute("ALTER TABLE training_result DROP COLUMN IF EXISTS user_id")
            await conn.execute("ALTER TABLE training_result DROP COLUMN IF EXISTS files_minio_key")
            logger.info("Migrated old columns and dropped duplicates from training_result")

    logger.info("Startup DB migration completed successfully")



async def insert_training_result(
    *,
    task_type: str,
    target: Optional[str],
    overall_train_start: datetime,
    overall_train_finish: datetime,
    status: str = "completed",
) -> int:
    """Вставляет новую запись в training_result и возвращает её training_id."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO training_result
                    (task_type, target, overall_train_start, overall_train_finish, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING training_id
                """,
                (task_type, target, overall_train_start, overall_train_finish, status),
            )
            row = await cur.fetchone()
            if not row:
                raise RuntimeError("Failed to insert training result")
            return row[0]


async def update_training_result(
    *,
    training_id: int,
    task_type: str,
    target: Optional[str],
    overall_train_finish: datetime,
    status: str = "completed",
) -> None:
    """Обновляет запись training_result при повторном обучении."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE training_result
            SET task_type = %s, target = %s, overall_train_finish = %s, status = %s
            WHERE training_id = %s
            """,
            (task_type, target, overall_train_finish, status, training_id),
        )


async def get_training_result(training_id: int) -> Optional[dict]:
    """Возвращает запись training_result по training_id или None."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT tr.training_id, ac.user_id, tr.task_type, tr.target,
                       ac.files_minio_key, tr.overall_train_start, tr.overall_train_finish, tr.status
                FROM training_result tr
                LEFT JOIN automl_chats ac ON tr.training_id = ac.training_id
                WHERE tr.training_id = %s
                """,
                (training_id,),
            )
            row = await cur.fetchone()
    if not row:
        return None
    return {
        "training_id": row[0],
        "user_id": row[1],
        "task_type": row[2],
        "target": row[3],
        "files_minio_key": row[4],
        "overall_train_start": row[5],
        "overall_train_finish": row[6],
        "status": row[7] if len(row) > 7 else "completed",
    }


async def insert_model_metadata(
    *,
    model_name: str,
    hyperparams: Optional[dict],
    metrics: Optional[dict],
    graphics: Optional[dict],
    place_in_training_res_batch: Optional[int],
    model_metadata: Optional[dict],
    train_start: datetime,
    train_finish: datetime,
) -> int:
    """Вставляет запись в models_metadata и возвращает model_id."""
    t_start = datetime.fromisoformat(train_start) if isinstance(train_start, str) else train_start
    t_finish = datetime.fromisoformat(train_finish) if isinstance(train_finish, str) else train_finish

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO models_metadata
                    (model_name, hyperparameters, metrics, graphics,
                     place_in_training_batch, model_metadata, train_start, train_finish)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING model_id
                """,
                (
                    model_name,
                    json.dumps(hyperparams) if hyperparams is not None else None,
                    json.dumps(metrics) if metrics is not None else None,
                    json.dumps(graphics) if graphics is not None else None,
                    place_in_training_res_batch,
                    json.dumps(model_metadata) if model_metadata is not None else None,
                    t_start,
                    t_finish,
                ),
            )
            row = await cur.fetchone()
            if not row:
                raise RuntimeError("Failed to insert model metadata")
            return row[0]


async def insert_training_x_model(*, training_id: int, model_id: int) -> int:
    """Вставляет запись в training_x_models и возвращает pair_id."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO training_x_models (training_id, model_id)
                VALUES (%s, %s)
                RETURNING pair_id
                """,
                (training_id, model_id),
            )
            row = await cur.fetchone()
            return row[0] if row else -1



async def get_history(user_id: int) -> list[dict]:
    """
    Возвращает историю обучений пользователя из таблицы automl_chats,
    объединенную с training_result для получения статуса и типа задачи.
    Отсортировано: сначала активные обучения, затем по времени обновления updated_at.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 
                    ac.chat_id, 
                    ac.dataset_name, 
                    ac.files_minio_key, 
                    ac.training_id, 
                    ac.created_at, 
                    ac.updated_at,
                    tr.task_type, 
                    tr.target, 
                    tr.status
                FROM automl_chats ac
                LEFT JOIN training_result tr ON ac.training_id = tr.training_id
                WHERE ac.user_id = %s
                ORDER BY CASE WHEN tr.status = 'training' THEN 1 ELSE 0 END DESC, ac.updated_at DESC
                """,
                (user_id,),
            )
            rows = await cur.fetchall()

    result = []
    for row in rows:
        chat_id, dataset_name, files_minio_key, training_id, created_at, updated_at, task_type, target, status = row
        
        result.append({
            "id": chat_id,
            "datasetName": dataset_name,
            "filename": dataset_name,
            "task_type": task_type,
            "target": target,
            "files_minio_key": files_minio_key,
            "created_at": updated_at.isoformat() if updated_at else None,
            "status": status or "idle",
            "training_id": str(training_id) if training_id else None,
        })
    return result


async def delete_training_result(training_id: int) -> bool:
    """
    Каскадно удаляет все данные, связанные с тренировкой:
    1. Находит все model_id в training_x_models для этой тренировки.
    2. Удаляет связи из training_x_models.
    3. Удаляет подробную информацию о моделях из models_metadata (освобождая дисковое пространство!).
    4. Удаляет основную запись из training_result.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # 1. Получаем все model_id, связанные с этим training_id
            await cur.execute(
                "SELECT model_id FROM training_x_models WHERE training_id = %s",
                (training_id,),
            )
            model_rows = await cur.fetchall()
            model_ids = [r[0] for r in model_rows]

            # 2. Удаляем связанные записи из связующей таблицы
            await cur.execute(
                "DELETE FROM training_x_models WHERE training_id = %s",
                (training_id,),
            )

            # 3. Удаляем модели из models_metadata
            if model_ids:
                q = sql.SQL("DELETE FROM models_metadata WHERE model_id IN ({})").format(
                    sql.SQL(", ").join(sql.Placeholder() * len(model_ids))
                )
                await cur.execute(q, tuple(model_ids))

            # 4. Теперь удаляем основную запись
            await cur.execute(
                "DELETE FROM training_result WHERE training_id = %s RETURNING training_id",
                (training_id,),
            )
            row = await cur.fetchone()
    return row is not None


async def insert_automl_chat(
    *,
    chat_id: str,
    user_id: int,
    dataset_name: str,
    files_minio_key: str,
) -> None:
    """Вставляет новую запись чата в automl_chats."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO automl_chats (chat_id, user_id, dataset_name, files_minio_key)
            VALUES (%s, %s, %s, %s)
            """,
            (chat_id, user_id, dataset_name, files_minio_key),
        )


async def update_automl_chat_training(
    *,
    chat_id: str,
    training_id: int,
) -> None:
    """Привязывает training_id к чату и обновляет updated_at."""
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE automl_chats
            SET training_id = %s, updated_at = NOW()
            WHERE chat_id = %s
            """,
            (training_id, chat_id),
        )


async def get_automl_chat(chat_id: str) -> Optional[dict]:
    """Возвращает информацию о чате по его chat_id."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT chat_id, user_id, dataset_name, files_minio_key, training_id, created_at, updated_at
                FROM automl_chats
                WHERE chat_id = %s
                """,
                (chat_id,),
            )
            row = await cur.fetchone()
    if not row:
        return None
    return {
        "chat_id": row[0],
        "user_id": row[1],
        "dataset_name": row[2],
        "files_minio_key": row[3],
        "training_id": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


async def delete_automl_chat(chat_id: str) -> bool:
    """Удаляет запись чата из automl_chats. Возвращает True если запись была найдена."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM automl_chats WHERE chat_id = %s RETURNING chat_id",
                (chat_id,),
            )
            row = await cur.fetchone()
    return row is not None


async def delete_untrained_chats_by_user(user_id: int) -> None:
    """Удаляет все чаты пользователя, у которых обучение не начато (training_id IS NULL)."""
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM automl_chats WHERE user_id = %s AND training_id IS NULL",
            (user_id,),
        )


async def get_training_detail(training_id: int) -> Optional[dict]:
    """
    Возвращает полную информацию о тренировке:
    заголовок + все связанные модели через training_x_models.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Основная запись
            await cur.execute(
                """
                SELECT tr.training_id, ac.files_minio_key, tr.task_type, tr.target, tr.status
                FROM training_result tr
                LEFT JOIN automl_chats ac ON tr.training_id = ac.training_id
                WHERE tr.training_id = %s
                """,
                (training_id,),
            )
            tr_row = await cur.fetchone()
            if not tr_row:
                return None

            # Все связанные модели
            await cur.execute(
                """
                SELECT mm.model_id, mm.model_name, mm.hyperparameters, mm.metrics,
                       mm.graphics, mm.model_metadata
                FROM training_x_models txm
                JOIN models_metadata mm ON txm.model_id = mm.model_id
                WHERE txm.training_id = %s
                """,
                (training_id,),
            )
            model_rows = await cur.fetchall()

    training_id_val, files_minio_key, task_type, target, status = tr_row

    # Восстанавливаем имя файла из ключа MinIO
    if files_minio_key and "/" in files_minio_key:
        filename = files_minio_key.split("/")[-1]
    else:
        filename = files_minio_key or f"Dataset_{training_id}"

    models_dict: dict[str, Any] = {}
    for m_row in model_rows:
        model_id, model_name, hyperparams, metrics, graphics, model_meta = m_row
        model_data: dict[str, Any] = {
            "model_name": model_name,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "graphics": graphics,
        }
        if model_meta:
            for k, v in model_meta.items():
                model_data[k] = v
        models_dict[model_name] = model_data

    return {
        "training_id": training_id_val,
        "files_minio_key": files_minio_key,
        "filename": filename,
        "task_type": task_type,
        "target": target,
        "status": status,
        "models": models_dict,
    }
