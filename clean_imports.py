import os
import subprocess

# Папки, которые не трогаем
exclude_dirs = {"__pycache__", "catboost_info", "datasets", "venv"}

# Список файлов, которые были изменены
changed_files = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            # Используем autoflake для очистки импортов на месте
            result = subprocess.run([
                "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                path
            ])
            if result.returncode == 0:
                changed_files.append(path)

# Обновляем индекс Git для изменённых файлов
if changed_files:
    subprocess.run(["git", "add"] + changed_files)
    print(f"Cleaned and staged {len(changed_files)} files.")