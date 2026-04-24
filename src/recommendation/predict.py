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
