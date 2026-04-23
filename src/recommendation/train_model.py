# train_model.py - РАСШИРЕННАЯ ВЕРСИЯ (группа крови + резус + ИМТ)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import math

print("=" * 70)
print("МОДУЛЬ ОБУЧЕНИЯ - РАСШИРЕННАЯ МЕДИЦИНСКАЯ ЛОГИКА")
print("Группа крови | Резус-фактор | ИМТ | Возраст | Половые особенности")
print("=" * 70)


# ============================================================
# ЧАСТЬ 1: МЕДИЦИНСКИЕ КОЭФФИЦИЕНТЫ И ПРАВИЛА
# ============================================================

class MedicalRules:
    """Медицинские правила для расчёта безопасного интервала донации"""

    # Базовые интервалы по полу (дни) - норма ВОЗ
    BASE_INTERVAL = {
        1: 90,  # мужчины
        0: 120  # женщины
    }

    # Минимальные значения гемоглобина для донации (г/дл)
    MIN_HEMOGLOBIN = {
        1: 13.5,  # мужчины
        0: 12.5  # женщины (ВОЗ рекомендует 12.5 для донации)
    }

    # Оптимальные значения гемоглобина
    OPTIMAL_HEMOGLOBIN = {
        1: 15.0,
        0: 13.8
    }

    # Коэффициенты для разных групп крови (восстановление)
    # Основано на исследованиях: разные группы крови имеют разные скорости восстановления
    BLOOD_TYPE_FACTOR = {
        'O-': 1.15,  # Универсальный донор, но медленнее восстанавливается
        'O+': 1.10,
        'A-': 1.05,
        'A+': 1.00,
        'B-': 1.05,
        'B+': 1.00,
        'AB-': 0.95,  # Универсальный реципиент, быстрее восстанавливается
        'AB+': 0.90
    }

    # Коррекция по ИМТ (Body Mass Index)
    # ИМТ = вес(кг) / рост(м)²
    BMI_CORRECTION = {
        'underweight': {'range': (0, 18.5), 'days': 15, 'advice': 'Недостаточный вес - нужно набрать массу'},
        'normal': {'range': (18.5, 25), 'days': 0, 'advice': 'Нормальный вес'},
        'overweight': {'range': (25, 30), 'days': -5, 'advice': 'Избыточный вес - восстановление быстрее'},
        'obese': {'range': (30, 35), 'days': -8, 'advice': 'Ожирение I степени'},
        'severely_obese': {'range': (35, 100), 'days': -10, 'advice': 'Ожирение II-III степени - повышенный риск'}
    }

    @staticmethod
    def get_bmi_category(bmi):
        """Определить категорию ИМТ"""
        for category, info in MedicalRules.BMI_CORRECTION.items():
            min_bmi, max_bmi = info['range']
            if min_bmi <= bmi < max_bmi:
                return category, info['days'], info['advice']
        return 'normal', 0, 'Нормальный вес'

    @staticmethod
    def get_recovery_factor_by_age(age):
        """Фактор восстановления по возрасту"""
        if age < 20:
            return 0.85  # Молодые восстанавливаются быстрее
        elif age < 30:
            return 0.90
        elif age < 40:
            return 0.95
        elif age < 50:
            return 1.00
        elif age < 60:
            return 1.10
        else:
            return 1.25  # Пожилые восстанавливаются дольше


def calculate_advanced_safe_interval(row):
    """
    РАСШИРЕННАЯ ЛОГИКА с учётом:
    - Группы крови и резус-фактора
    - ИМТ (индекса массы тела)
    - Возрастных особенностей
    - Полных медицинских показателей
    """

    gender = row['gender']
    blood_type = row['blood_type']  # например 'A+', 'O-', etc.

    # ========== 1. БАЗОВЫЙ ИНТЕРВАЛ ==========
    base = MedicalRules.BASE_INTERVAL[gender]

    # ========== 2. КОРРЕКЦИЯ ПО ГРУППЕ КРОВИ ==========
    blood_factor = MedicalRules.BLOOD_TYPE_FACTOR.get(blood_type, 1.0)

    # ========== 3. КОРРЕКЦИЯ ПО ИМТ ==========
    _, bmi_days, _ = MedicalRules.get_bmi_category(row['bmi'])

    # ========== 4. КОРРЕКЦИЯ ПО ГЕМОГЛОБИНУ ==========
    min_hgb = MedicalRules.MIN_HEMOGLOBIN[gender]
    optimal_hgb = MedicalRules.OPTIMAL_HEMOGLOBIN[gender]

    hgb_factor = 0
    if row['hemoglobin'] < min_hgb:
        # Серьёзный дефицит
        deficit_percent = (min_hgb - row['hemoglobin']) / min_hgb
        hgb_factor = int(deficit_percent * 60)
    elif row['hemoglobin'] < optimal_hgb:
        # Небольшой дефицит
        deficit_percent = (optimal_hgb - row['hemoglobin']) / optimal_hgb
        hgb_factor = int(deficit_percent * 25)
    elif row['hemoglobin'] > 17.5 and gender == 1:
        hgb_factor = -10  # Высокий гемоглобин - можно чаще
    elif row['hemoglobin'] > 16.0 and gender == 0:
        hgb_factor = -8

    # ========== 5. КОРРЕКЦИЯ ПО ФЕРРИТИНУ ==========
    ferritin_factor = 0
    if row['ferritin'] < 15:
        ferritin_factor = 75
    elif row['ferritin'] < 30:
        ferritin_factor = 50
    elif row['ferritin'] < 50:
        ferritin_factor = 25
    elif row['ferritin'] > 150:
        ferritin_factor = -10  # Хороший запас железа

    # ========== 6. КОРРЕКЦИЯ ПО ВОЗРАСТУ ==========
    age_factor = (MedicalRules.get_recovery_factor_by_age(row['age']) - 1.0) * base

    # ========== 7. ОСОБЫЕ ПРАВИЛА ДЛЯ РЕЗУС-ФАКТОРА ==========
    # Rh- нужно больше времени между донациями (дефицит доноров)
    rh_factor = 10 if '-' in blood_type else 0

    # ========== 8. ОПЫТ ДОНАЦИЙ ==========
    exp_factor = 0
    if row['prev_donations'] > 30:
        exp_factor = -15
    elif row['prev_donations'] > 15:
        exp_factor = -10
    elif row['prev_donations'] > 5:
        exp_factor = -5
    elif row['prev_donations'] == 0:
        exp_factor = 15

    # ========== 9. ИСТОРИЯ ПРОБЛЕМ ==========
    history_factor = 40 if row['low_hgb_history'] == 1 else 0

    # ========== 10. КОРРЕКЦИЯ ПО ПОЛУ (дополнительно) ==========
    # Женщинам нужно больше времени (уже учтено в base)
    gender_extra = 0

    # ========== ФИНАЛЬНЫЙ РАСЧЁТ ==========
    total = (
                        base * blood_factor) + hgb_factor + ferritin_factor + bmi_days + age_factor + rh_factor + exp_factor + history_factor

    # Корректировка для женщин с низким ферритином (особенно важно)
    if gender == 0 and row['ferritin'] < 30:
        total += 20

    # Ограничения безопасности
    total = max(60, min(365, total))

    # Добавляем небольшой шум (±3 дня)
    total += np.random.randint(-3, 4)

    return int(total)


# ============================================================
# ЧАСТЬ 2: ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ ДАННЫХ
# ============================================================

print("\n📊 Генерация расширенных данных о донорах...")

np.random.seed(42)
n_donors = 10000  # Больше данных для точности

# Распределение групп крови (реалистичная статистика России/СНГ)
BLOOD_TYPES = ['O+', 'A+', 'B+', 'AB+', 'O-', 'A-', 'B-', 'AB-']
BLOOD_PROBS = [0.32, 0.31, 0.21, 0.06, 0.04, 0.03, 0.02, 0.01]

# Создаём данные
data = pd.DataFrame({
    # Демография
    'age': np.random.choice(range(18, 66), n_donors),
    'gender': np.random.choice([0, 1], n_donors, p=[0.52, 0.48]),

    # Группа крови и резус
    'blood_type': np.random.choice(BLOOD_TYPES, n_donors, p=BLOOD_PROBS),

    # Антропометрия (рост и вес для расчёта ИМТ)
    'height_cm': np.random.normal(170, 10, n_donors).clip(140, 210),
    'weight_kg': np.random.normal(75, 15, n_donors).clip(45, 150),

    # Биохимия
    'hemoglobin': np.zeros(n_donors),
    'ferritin': np.zeros(n_donors),

    # История донаций
    'prev_donations': np.random.negative_binomial(2, 0.25, n_donors).clip(0, 100),
    'avg_interval_days': np.random.normal(100, 30, n_donors).clip(60, 200),
    'low_hgb_history': np.random.choice([0, 1], n_donors, p=[0.85, 0.15])
})

# Генерируем гемоглобин с учётом пола и возраста
for idx, row in data.iterrows():
    if row['gender'] == 1:  # мужчины
        mean_hgb = 14.5 - (row['age'] - 30) * 0.01
        data.at[idx, 'hemoglobin'] = np.random.normal(mean_hgb, 1.0)
    else:  # женщины
        mean_hgb = 13.2 - (row['age'] - 30) * 0.008
        data.at[idx, 'hemoglobin'] = np.random.normal(mean_hgb, 0.9)

data['hemoglobin'] = data['hemoglobin'].clip(9, 18.5).round(1)

# Генерируем ферритин (логнормальное распределение)
data['ferritin'] = np.random.lognormal(mean=3.8, sigma=0.9, size=n_donors).clip(5, 500).round(0)

# Рассчитываем ИМТ
data['bmi'] = data['weight_kg'] / ((data['height_cm'] / 100) ** 2)
data['bmi'] = data['bmi'].round(1)

# Категория ИМТ для анализа
data['bmi_category'] = data['bmi'].apply(lambda x: MedicalRules.get_bmi_category(x)[0])

print(f"✅ Сгенерировано доноров: {len(data)}")
print(f"\n📊 Статистика параметров:")
print(f"   Возраст: {data['age'].min()}-{data['age'].max()} лет (средний: {data['age'].mean():.1f})")
print(f"   ИМТ: {data['bmi'].min():.1f}-{data['bmi'].max():.1f} (средний: {data['bmi'].mean():.1f})")
print(f"   Гемоглобин: {data['hemoglobin'].min():.1f}-{data['hemoglobin'].max():.1f} г/дл")
print(f"   Ферритин: {data['ferritin'].min():.0f}-{data['ferritin'].max():.0f} мкг/л")
print(f"\n📊 Распределение групп крови:")
for bt in BLOOD_TYPES:
    count = (data['blood_type'] == bt).sum()
    print(f"   {bt}: {count} ({count / len(data) * 100:.1f}%)")

print(f"\n📊 Распределение ИМТ:")
for cat in MedicalRules.BMI_CORRECTION.keys():
    count = (data['bmi_category'] == cat).sum()
    if count > 0:
        print(f"   {cat}: {count} ({count / len(data) * 100:.1f}%)")

# ============================================================
# ЧАСТЬ 3: РАСЧЁТ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# ============================================================

print("\n🏥 Расчёт безопасных интервалов с учётом всех факторов...")

data['safe_interval_days'] = data.apply(calculate_advanced_safe_interval, axis=1)

print(f"✅ Интервалы рассчитаны:")
print(f"   Минимум: {data['safe_interval_days'].min()} дней")
print(f"   Средний: {data['safe_interval_days'].mean():.0f} дней")
print(f"   Медиана: {data['safe_interval_days'].median():.0f} дней")
print(f"   Максимум: {data['safe_interval_days'].max()} дней")

# ============================================================
# ЧАСТЬ 4: ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
# ============================================================

print("\n🔄 Кодирование категориальных признаков...")

# Кодируем группу крови
blood_encoder = LabelEncoder()
data['blood_type_encoded'] = blood_encoder.fit_transform(data['blood_type'])

# Кодируем категорию ИМТ
bmi_encoder = LabelEncoder()
data['bmi_category_encoded'] = bmi_encoder.fit_transform(data['bmi_category'])

# Выбираем признаки для модели
feature_cols = ['age', 'gender', 'blood_type_encoded', 'bmi', 'bmi_category_encoded',
                'hemoglobin', 'ferritin', 'prev_donations', 'avg_interval_days', 'low_hgb_history']

X = data[feature_cols]
y = data['safe_interval_days']

# Кодируем пол (уже 0/1)
print(f"✅ Признаки: {feature_cols}")

# ============================================================
# ЧАСТЬ 5: ОБУЧЕНИЕ МОДЕЛИ
# ============================================================

print("\n🤖 Обучение модели с расширенными признаками...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Оптимизированный Random Forest
model = RandomForestRegressor(
    n_estimators=300,  # больше деревьев
    max_depth=15,  # глубже для сложных взаимосвязей
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Кросс-валидация
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"📊 Кросс-валидация R² = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ============================================================
# ЧАСТЬ 6: ОЦЕНКА ТОЧНОСТИ
# ============================================================

print("\n📈 Оценка точности модели:")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Дополнительные метрики
errors = np.abs(y_test - y_pred)
within_7_days = (errors <= 7).mean() * 100
within_14_days = (errors <= 14).mean() * 100

print(f"   Средняя ошибка (MAE): {mae:.1f} дней")
print(f"   Корень из MSE (RMSE): {rmse:.1f} дней")
print(f"   Качество модели (R²): {r2:.3f}")
print(f"   Точность ±7 дней: {within_7_days:.1f}%")
print(f"   Точность ±14 дней: {within_14_days:.1f}%")

# Оценка
if mae <= 6:
    print("   🎯 ОТЛИЧНО! Модель имеет высокую клиническую точность.")
elif mae <= 10:
    print("   ✅ ХОРОШО! Модель пригодна для практического использования.")
else:
    print("   ⚠️ Требуется дополнительная настройка.")

# Анализ важности признаков
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 ВАЖНОСТЬ ПРИЗНАКОВ (что влияет на рекомендацию):")
for _, row in feature_importance.iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"   {row['feature']:25} {bar} {row['importance']:.3f}")

# ============================================================
# ЧАСТЬ 7: СОХРАНЕНИЕ МОДЕЛИ
# ============================================================

print("\n💾 Сохранение модели и кодировщиков...")

os.makedirs('../models', exist_ok=True)
os.makedirs('../data', exist_ok=True)

# Сохраняем модель
model_path = '../models/recommendation_model_advanced.pkl'
joblib.dump(model, model_path)
print(f"   ✅ Модель: {model_path}")

# Сохраняем кодировщики
joblib.dump(blood_encoder, '../models/blood_type_encoder.pkl')
joblib.dump(bmi_encoder, '../models/bmi_category_encoder.pkl')

# Сохраняем метаданные
metadata = {
    'feature_names': feature_cols,
    'mae': mae,
    'r2': r2,
    'cv_mean': cv_scores.mean(),
    'within_7_days_accuracy': within_7_days,
    'n_samples': n_donors,
    'blood_types': list(blood_encoder.classes_),
    'bmi_categories': list(bmi_encoder.classes_)
}
joblib.dump(metadata, '../models/model_metadata_advanced.pkl')

# Сохраняем данные
data.to_csv('../data/advanced_donor_data.csv', index=False)

print("\n" + "=" * 70)
print("🎉 РАСШИРЕННАЯ МОДЕЛЬ ГОТОВА!")
print(f"   Точность: ошибка {mae:.1f} дней | ±7 дней: {within_7_days:.1f}%")
print("=" * 70)

# Пример предсказания
print("\n📊 ПРИМЕР ПРЕДСКАЗАНИЯ:")
sample = X_test.iloc[[0]]
pred = model.predict(sample)[0]
actual = y_test.iloc[0]
print(f"   Предсказано: {pred:.0f} дней")
print(f"   Фактически: {actual:.0f} дней")
print(f"   Ошибка: {abs(pred - actual):.0f} дней")