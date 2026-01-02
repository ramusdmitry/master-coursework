"""
Скрипт для создания первого администратора
Использование: python scripts/create_admin.py --username admin --password admin123
"""
import argparse
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, engine
from app import models
from app.auth import get_password_hash

# Создание таблиц
models.Base.metadata.create_all(bind=engine)


def create_admin(username: str, password: str):
    """Создание администратора"""
    db = SessionLocal()
    try:
        # Проверка существования пользователя
        existing_user = db.query(models.User).filter(models.User.username == username).first()
        if existing_user:
            print(f"Пользователь {username} уже существует!")
            return False
        
        # Создание нового администратора
        hashed_password = get_password_hash(password)
        admin_user = models.User(
            username=username,
            hashed_password=hashed_password,
            is_admin=True
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print(f"Администратор {username} успешно создан!")
        return True
    except Exception as e:
        print(f"Ошибка при создании администратора: {e}")
        db.rollback()
        return False
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание администратора")
    parser.add_argument("--username", required=True, help="Имя пользователя")
    parser.add_argument("--password", required=True, help="Пароль")
    
    args = parser.parse_args()
    
    create_admin(args.username, args.password)


