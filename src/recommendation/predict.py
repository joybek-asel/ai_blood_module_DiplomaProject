# predict.py - РАСШИРЕННАЯ ВЕРСИЯ (группа крови + ИМТ)

import joblib
import numpy as np
import os

# Глобальные объекты
_model = None
_metadata = None
_blood_encoder = None
_bmi_encoder = None


def load_advanced_model():
    """Загружает расширенную модель и все компоненты"""
    global _model, _metadata, _blood_encoder, _bmi_encoder

    if _model is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, '..', '..', 'models')

        _model = joblib.load(os.path.join(models_path, 'recommendation_model_advanced.pkl'))
        _metadata = joblib.load(os.path.join(models_path, 'model_metadata_advanced.pkl'))
        _blood_encoder = joblib.load(os.path.join(models_path, 'blood_type_encoder.pkl'))
        _bmi_encoder = joblib.load(os.path.join(models_path, 'bmi_category_encoder.pkl'))

        print(f"✅ Загружена расширенная модель (точность: {_metadata['mae']:.1f} дней)")

    return _model, _metadata, _blood_encoder, _bmi_encoder


def calculate_bmi(weight_kg, height_cm):
    """Рассчитать ИМТ"""
    return weight_kg / ((height_cm / 100) ** 2)


def get_bmi_category(bmi):
    """Определить категорию ИМТ"""
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    elif bmi < 35:
        return 'obese'
    else:
        return 'severely_obese'


def get_recommendation_advanced(
        age: int,
        gender: int,
        blood_type: str,  # 'A+', 'O-', 'B+', etc.
        height_cm: float,  # рост в см
        weight_kg: float,  # вес в кг
        hemoglobin: float,
        ferritin: float,
        prev_donations: int,
        avg_interval_days: int,
        low_hgb_history: int
):
    """
    Расширенная рекомендация с учётом группы крови и ИМТ
    """
    model, metadata, blood_encoder, bmi_encoder = load_advanced_model()

    # Рассчитываем ИМТ
    bmi = calculate_bmi(weight_kg, height_cm)
    bmi_category = get_bmi_category(bmi)

    # Кодируем категории
    try:
        blood_encoded = blood_encoder.transform([blood_type])[0]
    except ValueError:
        raise ValueError(f"Неизвестная группа крови: {blood_type}. Допустимые: {metadata['blood_types']}")

    try:
        bmi_category_encoded = bmi_encoder.transform([bmi_category])[0]
    except ValueError:
        bmi_category_encoded = 1  # default to normal

    # Подготавливаем признаки
    features = np.array([[
        age, gender, blood_encoded, bmi, bmi_category_encoded,
        hemoglobin, ferritin, prev_donations, avg_interval_days, low_hgb_history
    ]])

    # Предсказание
    safe_days = int(model.predict(features)[0])

    # ========== ГЕНЕРАЦИЯ СОВЕТОВ ==========
    advice = []
    readiness = "green"

    # Совет по группе крови
    blood_advice = {
        'O-': "🩸 Вы универсальный донор (O-). Ваша кровь особо ценна!",
        'O+': "🩸 Вы донор O+, самый распространённый тип.",
        'AB+': "🩸 Вы универсальный реципиент (AB+)."
    }
    if blood_type in blood_advice:
        advice.append(blood_advice[blood_type])

    # Совет по ИМТ
    if bmi < 18.5:
        advice.append("⚠️ Недостаточный вес. Рекомендуем полноценное питание перед донацией.")
        readiness = "yellow"
    elif bmi > 30:
        advice.append("ℹ️ Избыточный вес. Контролируйте давление перед донацией.")

    # Совет по гемоглобину
    min_hgb = 13.5 if gender == 1 else 12.5
    if hemoglobin < min_hgb:
        advice.append(f"⚠️ Низкий гемоглобин ({hemoglobin} < {min_hgb}). Нужно восстановление.")
        readiness = "red"

    # Совет по ферритину
    if ferritin < 30:
        advice.append("⚠️ Низкий ферритин. Принимайте железо (после консультации с врачом).")
        if readiness != "red":
            readiness = "yellow"

    # Итоговый совет по интервалу
    if safe_days <= 90:
        advice.append("✅ Вы можете сдавать кровь в ближайшее время.")
    else:
        advice.append(f"🩸 Рекомендуемый перерыв: {safe_days} дней.")

    return {
        'next_donation_days': safe_days,
        'bmi': round(bmi, 1),
        'bmi_category': bmi_category,
        'health_advice': " ".join(advice),
        'ready_soon': safe_days <= 90 and readiness != 'red',
        'readiness_level': readiness
    }


# Демонстрация
if __name__ == "__main__":
    print("=" * 60)
    print("РАСШИРЕННЫЕ РЕКОМЕНДАЦИИ (группа крови + ИМТ)")
    print("=" * 60)

    result = get_recommendation_advanced(
        age=30, gender=1, blood_type='O-',
        height_cm=180, weight_kg=75,
        hemoglobin=15.2, ferritin=120,
        prev_donations=5, avg_interval_days=95, low_hgb_history=0
    )

    print(f"\n📊 Результат:")
    print(f"   ИМТ: {result['bmi']} ({result['bmi_category']})")
    print(f"   Через {result['next_donation_days']} дней")
    print(f"   {result['health_advice']}")
    print(f"   Статус: {result['readiness_level']}")

    

"""
# predict.py - ИСПРАВЛЕННАЯ ВЕРСИЯ для медицинской модели
# Модуль для получения персонализированных рекомендаций для донора

import joblib
import numpy as np
import os
import pandas as pd

# Глобальные переменные для загруженной модели (кеширование)
_model = None
_model_metadata = None


def load_model():
    """
    #Загружает сохранённую модель и метаданные
    """
    global _model, _model_metadata

    if _model is None:
        # Путь к папке models (относительно текущего файла)
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, '..', '..', 'models')

        model_path = os.path.join(models_path, 'recommendation_model_medical.pkl')
        metadata_path = os.path.join(models_path, 'model_metadata.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Сначала запустите train_model.py"
            )

        _model = joblib.load(model_path)

        # Загружаем метаданные (если есть)
        if os.path.exists(metadata_path):
            _model_metadata = joblib.load(metadata_path)
            print(f"✅ Загружена модель (точность: {_model_metadata.get('mae', '?')} дней)")
        else:
            _model_metadata = {'feature_names': ['age', 'gender', 'hemoglobin', 'ferritin',
                                                 'prev_donations', 'avg_interval_days', 'low_hgb_history']}
            print("✅ Загружена модель (без метаданных)")

    return _model, _model_metadata


def get_recommendation(
        age: int,
        gender: int,  # 0 = женщина, 1 = мужчина
        hemoglobin: float,  # г/дл
        ferritin: float,  # мкг/л
        prev_donations: int,
        avg_interval_days: int,
        low_hgb_history: int  # 0 = нет, 1 = да
):
    """
    #Возвращает персонализированную рекомендацию для донора

    #Returns:
    #    dict: {
    #        'next_donation_days': int,      # через сколько дней можно сдавать
    #        'health_advice': str,            # персональный совет по здоровью
    #        'ready_soon': bool,              # можно ли сдавать в ближайшие 90 дней
    #        'readiness_level': str,          # 'green' / 'yellow' / 'red'
    #        'confidence': float              # уверенность модели (0-1)
        }
    """

    # Загружаем модель
    model, metadata = load_model()

    # Проверка входных данных
    if not (18 <= age <= 65):
        print(f"⚠️ Внимание: возраст {age} выходит за пределы обучающих данных (18-65)")

    if gender not in [0, 1]:
        raise ValueError("gender должен быть 0 (женщина) или 1 (мужчина)")

    if hemoglobin <= 0 or hemoglobin > 20:
        raise ValueError(f"hemoglobin {hemoglobin} вне допустимого диапазона")

    if ferritin <= 0 or ferritin > 500:
        print(f"⚠️ Внимание: ферритин {ferritin} выходит за пределы обучающих данных")

    # Подготавливаем данные для предсказания
    features = np.array([[
        age,
        gender,
        hemoglobin,
        ferritin,
        prev_donations,
        avg_interval_days,
        low_hgb_history
    ]])

    # Делаем предсказание
    safe_days = int(model.predict(features)[0])

    # ------------------------------------------------------------
    # МЕДИЦИНСКАЯ ЛОГИКА для советов (улучшенная версия)
    # ------------------------------------------------------------
    health_advice = []
    readiness = "green"  # green = можно, yellow = осторожно, red = нельзя
    confidence = 0.85  # базовая уверенность (можно уточнить из метаданных)

    # 1. Проверка гемоглобина (самый важный фактор)
    if gender == 1:  # мужчина
        if hemoglobin < 12.0:
            health_advice.append("🔴 КРИТИЧЕСКИ низкий гемоглобин! Сдавать кровь НЕЛЬЗЯ. Обратитесь к врачу.")
            readiness = "red"
            safe_days = max(safe_days, 180)  # минимум полгода
        elif hemoglobin < 13.5:
            health_advice.append("⚠️ Гемоглобин ниже нормы. Рекомендуем: красное мясо, печень, гречку, яблоки.")
            readiness = "yellow"
        elif hemoglobin > 17.5:
            health_advice.append("ℹ️ Гемоглобин выше нормы. Проконсультируйтесь с врачом.")
    else:  # женщина
        if hemoglobin < 11.0:
            health_advice.append("🔴 КРИТИЧЕСКИ низкий гемоглобин! Сдавать кровь НЕЛЬЗЯ. Обратитесь к врачу.")
            readiness = "red"
            safe_days = max(safe_days, 180)
        elif hemoglobin < 12.0:
            health_advice.append("⚠️ Гемоглобин ниже нормы. Рекомендуем: красное мясо, печень, гречку, яблоки.")
            readiness = "yellow"
        elif hemoglobin > 15.5:
            health_advice.append("ℹ️ Гемоглобин выше нормы. Проконсультируйтесь с врачом.")

    # 2. Проверка ферритина (запасы железа)
    if ferritin < 15:
        health_advice.append("🔴 КРИТИЧЕСКИ низкий запас железа! Нужен перерыв 3-6 месяцев и консультация врача.")
        readiness = "red"
        safe_days = max(safe_days, 150)
    elif ferritin < 30:
        health_advice.append(
            "⚠️ Низкий запас железа (ферритин). Нужно восстановление: железосодержащие продукты/препараты.")
        if readiness != "red":
            readiness = "yellow"
    elif ferritin < 50:
        health_advice.append("ℹ️ Ферритин в пограничной зоне. Хорошо бы поднять уровень железа.")

    # 3. Совет по регулярности донаций
    if prev_donations == 0:
        health_advice.append("🌟 Вы впервые? Спасибо! Первая донация требует особого внимания к здоровью.")
    elif prev_donations > 20:
        health_advice.append("🏆 Вы опытный донор! Спасибо за регулярную помощь. Ваш организм адаптирован.")
        confidence = min(0.95, confidence + 0.05)

    # 4. Совет по интервалу (на основе предсказания)
    if safe_days <= 90:
        if readiness != "red":
            health_advice.append("✅ Вы можете планировать следующую донацию в ближайшее время.")
            readiness = "green" if readiness == "green" else readiness
    elif safe_days <= 120:
        health_advice.append("🩸 Рекомендуем подождать ~3-4 месяца перед следующей донацией.")
    elif safe_days <= 180:
        health_advice.append("⏰ Рекомендуется перерыв 4-6 месяцев для полного восстановления.")
    else:
        health_advice.append(
            "⚠️ Рекомендуется длительный перерыв (более 6 месяцев). Обязательно проконсультируйтесь с врачом.")
        readiness = "red"

    # 5. Возрастные рекомендации
    if age > 60:
        health_advice.append("👴 Для доноров старше 60 лет рекомендуем более частый контроль здоровья.")
    elif age < 20:
        health_advice.append("👶 Молодым донорам важно следить за питанием между донациями.")

    # 6. Учёт истории проблем
    if low_hgb_history == 1:
        health_advice.append("📋 У вас была история низкого гемоглобина. Следите за показателями особенно внимательно.")
        if readiness != "red":
            readiness = "yellow"

    # Объединяем советы в один текст
    advice_text = " ".join(health_advice)

    # Карта статусов для фронтенда
    readiness_map = {
        'green': '✅ Готов к донации',
        'yellow': '⚠️ Требуется осторожность',
        'red': '🔴 Донация НЕ рекомендуется'
    }

    return {
        'next_donation_days': safe_days,
        'health_advice': advice_text,
        'ready_soon': safe_days <= 90 and readiness != 'red',
        'readiness_level': readiness,
        'readiness_text': readiness_map[readiness],
        'confidence': confidence
    }


def get_batch_recommendations(donors_df):
    """
    #Получить рекомендации для нескольких доноров (batch prediction)

    #Args:
    #    donors_df: DataFrame с колонками как у модели

    #Returns:
    #    DataFrame с добавленными предсказаниями
    """
    model, metadata = load_model()

    # Убеждаемся, что колонки в правильном порядке
    feature_cols = metadata.get('feature_names',
                                ['age', 'gender', 'hemoglobin', 'ferritin',
                                 'prev_donations', 'avg_interval_days', 'low_hgb_history'])

    X = donors_df[feature_cols]
    predictions = model.predict(X)

    result_df = donors_df.copy()
    result_df['predicted_interval_days'] = predictions.astype(int)

    return result_df


# ------------------------------------------------------------
# ДЕМОНСТРАЦИЯ
# ------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ ДЛЯ ДОНОРА КРОВИ (Медицинская логика)")
    print("=" * 60)

    # Тестовые примеры
    test_cases = [
        {
            "name": "Здоровый мужчина",
            "params": {
                'age': 30, 'gender': 1, 'hemoglobin': 15.2,
                'ferritin': 120, 'prev_donations': 5,
                'avg_interval_days': 95, 'low_hgb_history': 0
            }
        },
        {
            "name": "Женщина с низким гемоглобином",
            "params": {
                'age': 25, 'gender': 0, 'hemoglobin': 11.2,
                'ferritin': 25, 'prev_donations': 2,
                'avg_interval_days': 110, 'low_hgb_history': 1
            }
        },
        {
            "name": "Проблемный случай (критический)",
            "params": {
                'age': 45, 'gender': 1, 'hemoglobin': 10.5,
                'ferritin': 10, 'prev_donations': 0,
                'avg_interval_days': 0, 'low_hgb_history': 1
            }
        }
    ]

    for case in test_cases:
        print(f"\n📋 {case['name']}:")
        result = get_recommendation(**case['params'])
        print(f"   🗓️  Через {result['next_donation_days']} дней")
        print(f"   📊 Статус: {result['readiness_text']}")
        print(f"   💡 {result['health_advice'][:100]}...")

    print("\n" + "=" * 60)
    print("💡 Функция готова к использованию!")
    print("=" * 60)"""