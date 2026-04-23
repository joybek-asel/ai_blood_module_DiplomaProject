# upload_to_postgres.py
# Модуль для загрузки данных доноров в PostgreSQL
# Расположение: src/recommendation/upload_to_postgres.py

import pandas as pd
import psycopg2
from psycopg2 import sql, OperationalError
from sqlalchemy import create_engine
import os
import sys
from datetime import datetime

# Добавляем родительскую директорию в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# НАСТРОЙКИ ПОДКЛЮЧЕНИЯ (ИЗМЕНИТЕ ПОД ВАШУ БД!)
# ============================================================

DB_CONFIG = {
    'host': 'localhost',  # или IP вашего сервера PostgreSQL
    'port': 5432,  # стандартный порт PostgreSQL
    'database': 'donor_db',
    'user': 'postgres',
    'password': '123'
}

# Альтернативный формат для SQLAlchemy
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"


# ============================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С БАЗОЙ ДАННЫХ
# ============================================================

def create_connection():
    """Создаёт соединение с PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Подключение к PostgreSQL установлено")
        return conn
    except OperationalError as e:
        print(f"❌ Ошибка подключения: {e}")
        print("\nПроверьте:")
        print("1. Запущен ли PostgreSQL сервер")
        print("2. Правильные ли хост, порт, логин, пароль")
        print("3. Существует ли база данных")
        return None


def create_tables(conn):
    """Создаёт необходимые таблицы в PostgreSQL"""
    cursor = conn.cursor()

    # Таблица доноров
    create_donors_table = """
    CREATE TABLE IF NOT EXISTS donors (
        id SERIAL PRIMARY KEY,
        age INT NOT NULL,
        gender INT NOT NULL,
        blood_type VARCHAR(3) NOT NULL,
        height_cm DECIMAL(5,1),
        weight_kg DECIMAL(5,1),
        bmi DECIMAL(4,1),
        hemoglobin DECIMAL(4,1) NOT NULL,
        ferritin DECIMAL(6,1),
        prev_donations INT DEFAULT 0,
        avg_interval_days INT,
        low_hgb_history INT DEFAULT 0,
        safe_interval_days INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Таблица рекомендаций (результаты ИИ)
    create_recommendations_table = """
    CREATE TABLE IF NOT EXISTS recommendations (
        id SERIAL PRIMARY KEY,
        donor_id INT REFERENCES donors(id) ON DELETE CASCADE,
        predicted_interval_days INT NOT NULL,
        readiness_level VARCHAR(10),
        health_advice TEXT,
        confidence DECIMAL(3,2),
        model_version VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Таблица для мониторинга качества модели
    create_model_metrics_table = """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id SERIAL PRIMARY KEY,
        model_version VARCHAR(50),
        mae DECIMAL(5,2),
        r2 DECIMAL(4,3),
        accuracy_7days DECIMAL(4,1),
        accuracy_14days DECIMAL(4,1),
        n_samples INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        cursor.execute(create_donors_table)
        cursor.execute(create_recommendations_table)
        cursor.execute(create_model_metrics_table)
        conn.commit()
        print("✅ Все таблицы созданы/проверены")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания таблиц: {e}")
        conn.rollback()
        return False


def load_data_from_csv(csv_path=None):
    """Загружает данные из CSV файла"""
    if csv_path is None:
        # Ищем файл с данными
        possible_paths = [
            '../data/advanced_donor_data.csv',
            '../../data/advanced_donor_data.csv',
            'data/advanced_donor_data.csv'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break

    if not csv_path or not os.path.exists(csv_path):
        print(f"❌ Файл не найден: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"✅ Загружено {len(df)} записей из {csv_path}")
    return df


def upload_donors_to_postgres(df, conn):
    """Загружает данные доноров в PostgreSQL"""
    cursor = conn.cursor()

    # SQL запрос для вставки
    insert_query = """
    INSERT INTO donors (
        age, gender, blood_type, height_cm, weight_kg, bmi,
        hemoglobin, ferritin, prev_donations, avg_interval_days,
        low_hgb_history, safe_interval_days
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO NOTHING
    """

    # Подготавливаем данные
    records = []
    for _, row in df.iterrows():
        # Проверяем наличие всех полей
        record = (
            int(row.get('age', 0)),
            int(row.get('gender', 0)),
            str(row.get('blood_type', 'O+')),
            float(row.get('height_cm', 170)) if pd.notna(row.get('height_cm')) else None,
            float(row.get('weight_kg', 70)) if pd.notna(row.get('weight_kg')) else None,
            float(row.get('bmi', 22)) if pd.notna(row.get('bmi')) else None,
            float(row.get('hemoglobin', 14)),
            float(row.get('ferritin', 50)) if pd.notna(row.get('ferritin')) else None,
            int(row.get('prev_donations', 0)),
            int(row.get('avg_interval_days', 90)) if pd.notna(row.get('avg_interval_days')) else None,
            int(row.get('low_hgb_history', 0)),
            int(row.get('safe_interval_days', 90)) if pd.notna(row.get('safe_interval_days')) else None
        )
        records.append(record)

    # Вставляем пакетами по 1000 записей
    batch_size = 1000
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            cursor.executemany(insert_query, batch)
            conn.commit()
            total_inserted += len(batch)
            print(f"   Загружено {total_inserted}/{len(records)} записей")
        except Exception as e:
            print(f"   Ошибка в пакете {i}: {e}")
            conn.rollback()

    print(f"✅ Загружено {total_inserted} доноров в PostgreSQL")
    return total_inserted


def save_model_metrics(conn, model_version, metrics):
    """Сохраняет метрики модели в БД"""
    cursor = conn.cursor()

    insert_query = """
    INSERT INTO model_metrics (model_version, mae, r2, accuracy_7days, accuracy_14days, n_samples)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    try:
        cursor.execute(insert_query, (
            model_version,
            metrics.get('mae', 0),
            metrics.get('r2', 0),
            metrics.get('within_7_days', 0),
            metrics.get('within_14_days', 0),
            metrics.get('n_samples', 0)
        ))
        conn.commit()
        print("✅ Метрики модели сохранены")
        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения метрик: {e}")
        return False


def test_connection():
    """Тестирует подключение к PostgreSQL"""
    print("\n🔍 Тест подключения к PostgreSQL...")
    conn = create_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"   PostgreSQL версия: {version[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    return False


# ============================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ В POSTGRESQL")
    print("=" * 60)

    # 1. Тестируем подключение
    if not test_connection():
        print("\n❌ Не удалось подключиться к PostgreSQL")
        print("\nЧтобы исправить:")
        print("1. Установите PostgreSQL: https://www.postgresql.org/download/")
        print("2. Создайте базу данных: CREATE DATABASE donor_db;")
        print("3. Обновите пароль в DB_CONFIG['password']")
        return

    # 2. Подключаемся
    conn = create_connection()
    if not conn:
        return

    # 3. Создаём таблицы
    if not create_tables(conn):
        conn.close()
        return

    # 4. Загружаем данные из CSV
    df = load_data_from_csv()
    if df is None:
        conn.close()
        return

    # 5. Загружаем в PostgreSQL
    print("\n📤 Загрузка данных в PostgreSQL...")
    uploaded = upload_donors_to_postgres(df, conn)

    # 6. Сохраняем метрики модели (если есть)
    try:
        import joblib
        metadata_path = '../models/model_metadata_advanced.pkl'
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            save_model_metrics(conn, 'advanced_v1', metadata)
    except:
        pass

    conn.close()

    print("\n" + "=" * 60)
    print(f"✅ ГОТОВО! Загружено {uploaded} доноров")
    print("=" * 60)

    # Выводим пример данных
    print("\n📊 Пример загруженных данных:")
    print(df[['age', 'gender', 'blood_type', 'hemoglobin', 'safe_interval_days']].head(10))


if __name__ == "__main__":
    main()