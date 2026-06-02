import pickle
import json
import re
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional, Callable

from fastapi import FastAPI, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import base64
import io
import uvicorn

from automl.models import AutoML, TaskType
from automl.config import (
    ClusteringConfig,
    ClassificationConfig,
    RegressionConfig,
)
from automl.cluster.utils import save_clustering_report
from automl.services.training_persistence import TrainingPersistenceService
from automl.funcs import compute_mixed_corr

from db.pool import pool
from db.queries import (
    run_startup_migrations,
    insert_training_result,
    update_training_result,
    get_training_result,
    get_training_detail,
    get_history,
    delete_training_result,
    insert_automl_chat,
    update_automl_chat_training,
    get_automl_chat,
    delete_automl_chat,
    delete_untrained_chats_by_user,
)

from redisclient import RedisClient
from setup.logger import logger
from setup.minio import upload_dataset_to_minio, download_dataset_from_minio


redis = RedisClient()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()
    logger.info("Пул соединений с БД открыт.")

    try:
        await run_startup_migrations()
    except Exception as e:
        logger.error(f"Startup DB migration error: {e}")

    yield

    await pool.close()
    logger.info("Пул соединений с БД закрыт.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:80",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(filename: str) -> str:
    """Убирает из имени файла всё кроме букв, цифр, точки, дефиса и подчёркивания."""
    name = filename.strip()
    name = re.sub(r"[^\w.\-]", "_", name)
    return name[:100]  # обрезаем чтобы ключ вмещался в VARCHAR(200)


def make_minio_key(user_id: int, filename: str) -> str:
    """
    Формирует ключ MinIO: '{user_id}/{folder_name}/{filename}'.
    Где folder_name — это filename без расширения.
    """
    import os
    folder_name, _ = os.path.splitext(filename)
    return f"{user_id}/{folder_name}/{filename}"


def parse_from_b64(file_b64: str) -> pd.DataFrame:
    file_bytes = base64.b64decode(file_b64)
    buf = io.BytesIO(file_bytes)
    try:
        df = pd.read_csv(buf)
    except Exception:
        buf.seek(0)
        df = pd.read_excel(buf)
    return df


def run_automl(
    task: str,
    target: Optional[str],
    df: pd.DataFrame,
    cols_to_remove: list[str],
    user_id: int,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> dict:
    logger.info(f"new request received | user_id={user_id} task={task} target={target}")
    automl = AutoML(logger=logger, random_state=42)

    try:
        if task == "clustering":
            task_enum = TaskType.CLUSTERING
            cfg = ClusteringConfig()
        elif task == "regression":
            task_enum = TaskType.REGRESSION
            cfg = RegressionConfig()
        else:
            task_enum = TaskType.CLASSIFICATION
            cfg = ClassificationConfig()

        training_result, preproc_results = automl.run(
            task_type=task_enum,
            df=df,
            target=target,
            config=cfg,
            cols_to_remove=cols_to_remove,
            preprocessing=False,
            progress_callback=progress_callback,
        )

        if task == "clustering":
            save_clustering_report(training_result=training_result, filepath="clustering_report.txt")

    except ValueError as e:
        return {"error": str(e)}

    return {
        "preprocess": preproc_results,
        "training": training_result,
    }


async def save_all_models_background(
    *,
    models: dict,
    training_id: int,
) -> None:
    """Сохраняет результаты моделей в БД после обучения."""
    persistence = TrainingPersistenceService(logger=logger)
    try:
        for i, model in enumerate(models.values(), start=1):
            await persistence.save_model_result(
                result=model,
                place_in_batch=i,
                new_training_id=training_id,
            )
        logger.info(f"finished saving models to db | training_id={training_id}")
    except Exception as e:
        logger.exception(f"background save error: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/corr")
async def get_corr(
    file_b64: str = Form(...),
    filename: str = Form(...),
    user_id: Optional[str] = Form("1"),
) -> JSONResponse:
    """
    1. Принимает сырой датасет (base64).
    2. Запускает предобработку.
    3. Сохраняет очищенный df в MinIO и регистрирует ключ в Redis под user_id.
    4. Возвращает корреляционную матрицу + data_id (== minio_key).
    НЕ создаёт запись training_result — это происходит только при /train.
    """
    logger.info(f"corr requested | user_id={user_id} filename={filename}")
    uid = int(user_id) if user_id else 1

    df_raw = parse_from_b64(file_b64)

    automl_inst = AutoML(logger=logger)
    _, _, df_corr, _, _ = automl_inst.preprocess(
        df=df_raw,
        task="classification",
        cols_to_remove=[],
    )

    corr_matrix, labels = compute_mixed_corr(df_corr)

    # Ключ MinIO: "{user_id}/{safe_filename}" — из него восстанавливается имя файла
    minio_key = make_minio_key(uid, filename)

    try:
        dataset_bytes = pickle.dumps(df_corr)
        await upload_dataset_to_minio(minio_key, dataset_bytes)

        # Сохраняем ключ в Redis под user_id (TTL = 15 мин)
        # Это перезаписывает предыдущий датасет юзера в кэше Redis (один файл одновременно)
        redis.set_user_dataset(uid, minio_key, filename)

        # Защита от спама: удаляем любой предыдущий чат пользователя, в котором еще не началось обучение
        await delete_untrained_chats_by_user(uid)

        # Регистрируем новый чат в automl_chats сразу после загрузки датасета
        chat_id = uuid.uuid4().hex
        await insert_automl_chat(
            chat_id=chat_id,
            user_id=uid,
            dataset_name=filename,
            files_minio_key=minio_key
        )

        return JSONResponse(
            {
                "corr_matrix": corr_matrix.tolist(),
                "labels": labels,
                "data_id": chat_id, # Возвращаем chat_id UUID в качестве data_id
            },
            status_code=200,
        )
    except Exception as e:
        logger.exception(f"Error in /corr: {e}")
        return JSONResponse(
            {
                "corr_matrix": corr_matrix.tolist(),
                "labels": labels,
                "data_id": None,
                "error": "Failed to persist dataset, you can still train but history won't be saved.",
            },
            status_code=200,
        )


async def background_train_flow(
    training_id: int,
    task: str,
    target: Optional[str],
    cols_to_remove: list[str],
    user_id: int,
    files_minio_key: str,
) -> None:
    """Выполняет скачивание датасета, обучение AutoML и сохранение результатов в фоне."""
    logger.info(f"[Background] Starting training_id={training_id} for user_id={user_id}...")
    try:
        # 1. Скачиваем очищенный датасет из MinIO
        redis.set_training_progress(training_id, "Загрузка данных", 0)
        dataset_bytes = await download_dataset_from_minio(files_minio_key)
        df = pickle.loads(dataset_bytes)

        # 2. Запускаем обучение
        overall_train_start = datetime.now(timezone.utc)
        def progress_cb(step_name: str, step_idx: int, model_name: Optional[str] = None):
            redis.set_training_progress(training_id, step_name, step_idx, model_name)

        res = await asyncio.to_thread(
            run_automl,
            task=task,
            target=target,
            df=df,
            cols_to_remove=cols_to_remove,
            user_id=user_id,
            progress_callback=progress_cb,
        )
        overall_train_finish = datetime.now(timezone.utc)

        if "error" in res:
            logger.error(f"[Background] AutoML run error for training_id={training_id}: {res['error']}")
            await update_training_result(
                training_id=training_id,
                task_type=task,
                target=target,
                overall_train_finish=overall_train_finish,
                status="failed",
            )
            return

        # 3. Сохраняем результат в БД и обновляем статус
        if "training" in res and "models" in res["training"]:
            # Сохраняем модели в БД СНАЧАЛА
            redis.set_training_progress(training_id, "Сохранение моделей", 5)
            persistence = TrainingPersistenceService(logger=logger)
            models = res["training"]["models"]
            for i, model in enumerate(models.values(), start=1):
                await persistence.save_model_result(
                    result=model,
                    place_in_batch=i,
                    new_training_id=training_id,
                )
            
            # Обновляем статус на completed ПОСЛЕ сохранения моделей
            await update_training_result(
                training_id=training_id,
                task_type=task,
                target=target,
                overall_train_finish=overall_train_finish,
                status="completed",
            )
            redis.set_training_progress(training_id, "Завершено", 6)
            logger.info(f"[Background] Success! Models saved for training_id={training_id}")
        else:
            logger.error(f"[Background] No models returned for training_id={training_id}")
            await update_training_result(
                training_id=training_id,
                task_type=task,
                target=target,
                overall_train_finish=overall_train_finish,
                status="failed",
            )

    except Exception as e:
        logger.exception(f"[Background] Exception in background_train_flow for training_id={training_id}: {e}")
        try:
            await update_training_result(
                training_id=training_id,
                task_type=task,
                target=target,
                overall_train_finish=datetime.now(timezone.utc),
                status="failed",
            )
        except Exception as db_err:
            logger.error(f"[Background] Failed to set status to failed in DB: {db_err}")


@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    data_id: str = Form(...),  # Обязательно: chat_id UUID
    task: str = Form(...),
    target: Optional[str] = Form(None),
    cols_to_remove: Optional[str] = Form(None),
    user_id: str = Form(...),
    training_id: Optional[str] = Form(None),  # Сохраняем для обратной совместимости
) -> JSONResponse:
    """
    Воркфлоу:
    - Достаём информацию о чате из automl_chats по data_id (chat_id UUID).
    - Если у чата уже есть training_id → это повторное обучение:
      обновляем статус существующего training_result на 'training'.
    - Иначе → первое обучение:
      создаём новую запись в training_result, привязываем её к чату и очищаем кэш в Redis.
    - Запускаем background_train_flow в фоне и возвращаем chat_id немедленно.
    """
    uid = int(user_id)

    if cols_to_remove:
        try:
            columns_to_remove: list[str] = json.loads(cols_to_remove)
            if not isinstance(columns_to_remove, list):
                raise ValueError
        except Exception:
            return JSONResponse(
                {"error": "cols_to_remove must be a JSON list of strings"},
                status_code=400,
            )
    else:
        columns_to_remove = []

    # ------------------------------------------------------------------
    # 1. Загружаем сессию чата из БД по chat_id
    # ------------------------------------------------------------------
    chat = await get_automl_chat(data_id)
    if not chat:
        return JSONResponse(
            {"error": f"AutoML Chat session with ID {data_id} not found in DB"},
            status_code=404,
        )

    files_minio_key = chat["files_minio_key"]
    filename = chat["dataset_name"]
    overall_train_start = datetime.now(timezone.utc)

    if chat["training_id"]:
        # Повторное обучение в рамках существующего чата
        existing_training_id = chat["training_id"]
        logger.info(f"Repeat training inside chat_id={data_id} | existing_training_id={existing_training_id}")
        
        # Сразу ставим статус 'training' в БД
        await update_training_result(
            training_id=existing_training_id,
            task_type=task,
            target=target,
            overall_train_finish=overall_train_start,
            status="training",
        )
        new_training_id = existing_training_id
    else:
        # Первое обучение в рамках этого чата
        logger.info(f"First training inside chat_id={data_id}")
        new_training_id = await insert_training_result(
            task_type=task,
            target=target,
            overall_train_start=overall_train_start,
            overall_train_finish=overall_train_start,
            status="training",
        )
        # Привязываем training_id к automl_chats
        await update_automl_chat_training(chat_id=data_id, training_id=new_training_id)
        
        # Чистим Redis сразу (после нажатия Обучить удаляем временный датасет)
        redis.delete_user_dataset(uid)

    # ------------------------------------------------------------------
    # 2. Добавляем фоновую задачу обучения
    # ------------------------------------------------------------------
    background_tasks.add_task(
        background_train_flow,
        training_id=new_training_id,
        task=task,
        target=target,
        cols_to_remove=columns_to_remove,
        user_id=uid,
        files_minio_key=files_minio_key,
    )

    return JSONResponse(
        {
            "message": "Training started in background",
            "training_id": new_training_id,
            "status": "training",
        },
        status_code=200,
    )


@app.get("/train/progress/{training_id}")
async def get_training_progress(training_id: int):
    """Стриминг прогресса обучения через Server-Sent Events (SSE) и Redis Pub/Sub."""
    
    async def event_generator():
        # 1. Отправляем текущее состояние сразу
        current = redis.get_training_progress(training_id)
        if current:
            yield f"data: {json.dumps(current)}\n\n"
            if current.get("step_index") == 6:
                return

        # 2. Подписываемся на канал Pub/Sub
        pubsub = redis.redis_client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(f"channel:training_progress:{training_id}")
        
        try:
            while True:
                # Проверяем наличие новых сообщений без блокировки
                msg = pubsub.get_message()
                if msg:
                    data = msg['data']
                    yield f"data: {data.decode('utf-8')}\n\n"
                    
                    # Если дошли до завершающего этапа (6) — завершаем стрим
                    try:
                        parsed = json.loads(data)
                        if parsed.get("step_index") == 6:
                            break
                    except Exception:
                        pass
                
                # Засыпаем на 0.5с для передачи контроля событийному циклу
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info(f"SSE client disconnected for training_id={training_id}")
        finally:
            pubsub.unsubscribe()
            pubsub.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/history")
async def get_history_endpoint(user_id: int) -> list:
    return await get_history(user_id)


@app.delete("/history/{chat_id}")
async def delete_history(chat_id: str):
    chat = await get_automl_chat(chat_id)
    if not chat:
        # Fallback to direct training delete if numeric id is passed
        if chat_id.isdigit():
            found = await delete_training_result(int(chat_id))
            if not found:
                return JSONResponse({"error": "Training result not found"}, status_code=404)
            return {"message": "Deleted successfully"}
        return JSONResponse({"error": "Chat session not found"}, status_code=404)
    
    if chat["training_id"]:
        await delete_training_result(chat["training_id"])
    
    found = await delete_automl_chat(chat_id)
    if not found:
        return JSONResponse({"error": "Failed to delete chat"}, status_code=500)
    return {"message": "Deleted successfully"}


@app.get("/history/{chat_id}")
async def get_history_detail(chat_id: str):
    chat = await get_automl_chat(chat_id)
    detail = None
    if not chat:
        # Fallback for backward compatibility with old numeric training_id
        if chat_id.isdigit():
            detail = await get_training_detail(int(chat_id))
            if not detail:
                return JSONResponse({"error": "Training result not found"}, status_code=404)
        else:
            return JSONResponse({"error": "Chat session not found"}, status_code=404)
        
        files_minio_key = detail["files_minio_key"]
        filename = detail["filename"]
    else:
        if chat["training_id"]:
            detail = await get_training_detail(chat["training_id"])
        
        files_minio_key = detail["files_minio_key"] if detail else chat["files_minio_key"]
        filename = detail["filename"] if detail else chat["dataset_name"]

    models_dict = detail.get("models", {}) if detail else {}
    result_data = (
        {"training": {"models": models_dict}}
        if models_dict
        else None
    )

    try:
        dataset_bytes = await download_dataset_from_minio(files_minio_key)
        # The stored DataFrame is already preprocessed (saved during /corr upload).
        df = pickle.loads(dataset_bytes)

        # Compute correlation directly on the stored (already clean) DataFrame
        corr_matrix, labels = compute_mixed_corr(df)

        records = json.loads(df.to_json(orient="records"))
        columns = df.columns.tolist()

        return {
            "training_id": str(detail["training_id"]) if (detail and detail.get("training_id")) else None,
            "filename": filename,
            "task_type": detail.get("task_type") if detail else None,
            "target": detail.get("target") if detail else None,
            "status": detail.get("status", "idle") if detail else "idle",
            "data": records,
            "columns": columns,
            "data_id": chat_id,  # We always return the active chat_id UUID as data_id
            "corr_matrix": corr_matrix.tolist(),
            "labels": labels,
            "result": result_data,
        }
    except Exception as e:
        logger.exception(f"Error loading dataset from MinIO: {e}")
        return JSONResponse({"error": f"Failed to load dataset: {str(e)}"}, status_code=500)


from fastapi.responses import StreamingResponse

@app.get("/download")
async def download_file_endpoint(
    minio_key: str,
    filename: str,
) -> Any:
    """
    Скачивает файл модели из MinIO по minio_key и отдаёт его в виде потока с оригинальным именем.
    """
    logger.info(f"download requested | key={minio_key} filename={filename}")
    try:
        file_bytes = await download_dataset_from_minio(minio_key)
        return StreamingResponse(
            io.BytesIO(file_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        logger.exception(f"Error downloading file {minio_key}: {e}")
        return JSONResponse({"error": f"Failed to download file: {str(e)}"}, status_code=500)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception during request {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
    )