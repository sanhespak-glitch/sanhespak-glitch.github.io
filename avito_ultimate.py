# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")

print("="*70)
print("АВИТО ХАКАТОН: АНАЛИЗ ЛИКВИДНОСТИ")
print("="*70)

print("\n[1/8] Загрузка данных...")
base = pd.read_csv(r"C:\Users\User\Downloads\jealous\liquidity_base.csv")
cnt = pd.read_csv(r"C:\Users\User\Downloads\jealous\liquidity_cnt.csv")
print(f"   Объявлений: {len(base):,}")
print(f"   Контактов: {len(cnt):,}")

print("\n[2/8] Очистка данных...")
base = base.drop_duplicates(subset=["id"])
base["start_time"] = pd.to_datetime(base["start_time"])
base["close_time"] = pd.to_datetime(base["close_time"])
base["lifetime_days"] = (base["close_time"] - base["start_time"]).dt.days.clip(lower=1)

for col in ["price", "mileage"]:
    q1 = base[col].quantile(0.01)
    q99 = base[col].quantile(0.99)
    base = base[(base[col] >= q1) & (base[col] <= q99)]

base = base[(base["year"] >= 1970) & (base["year"] <= 2024)]

total = cnt.groupby("id")["cnt_contacts"].sum().reset_index()
total.columns = ["id", "total_contacts"]
base = base.merge(total, on="id", how="left")
base["total_contacts"] = base["total_contacts"].fillna(0)
base["contacts_per_day"] = base["total_contacts"] / base["lifetime_days"]
base["car_age"] = 2024 - base["year"]
base["log_contacts"] = np.log1p(base["total_contacts"])
base["log_price"] = np.log1p(base["price"])
print(f"   После очистки: {len(base):,} объявлений")

# ВИЗУАЛИЗАЦИЯ 1: Корреляционная матрица
print("\n[3/8] Визуализация 1: Корреляционная матрица...")
num_cols = ["price", "year", "mileage", "total_contacts", "lifetime_days", "contacts_per_day"]
num_cols = [c for c in num_cols if c in base.columns]
corr = base[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Корреляционная матрица факторов ликвидности")
plt.tight_layout()
plt.savefig("01_correlation.png", dpi=150)
plt.show()
print("   Сохранено: 01_correlation.png")

print("\n[4/8] Сегментация...")
base["segment"] = "Другое"
base.loc[base["total_contacts"] > base["total_contacts"].quantile(0.8), "segment"] = "Высокая ликвидность"
base.loc[base["condition"] == "Битый", "segment"] = "Битые"
premium = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Land Rover", "Porsche"]
base.loc[base["brand"].isin(premium), "segment"] = "Премиум"
mass = ["Toyota", "Kia", "Hyundai", "Volkswagen", "Renault", "Skoda", "Ford", "Nissan", "LADA"]
base.loc[base["brand"].isin(mass), "segment"] = "Массовые"
base.loc[base["car_age"] > 15, "segment"] = "Возрастные >15 лет"
base.loc[base["price"] > 3000000, "segment"] = "Дорогие >3 млн"

seg_stats = base.groupby("segment")["total_contacts"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
print("\n   Статистика по сегментам:")
for seg in seg_stats.index:
    print(f"   • {seg}: {seg_stats.loc[seg]['count']} авто, среднее {seg_stats.loc[seg]['mean']:.1f} контактов")

# ВИЗУАЛИЗАЦИЯ 2: Boxplot сегментов
print("\n[5/8] Визуализация 2: Сравнение сегментов (boxplot)...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=base, x="segment", y="total_contacts")
plt.xticks(rotation=45, ha="right")
plt.title("Распределение контактов по сегментам")
plt.tight_layout()
plt.savefig("02_segments.png", dpi=150)
plt.show()
print("   Сохранено: 02_segments.png")

# ВИЗУАЛИЗАЦИЯ 3: Feature Importance
print("\n[6/8] Визуализация 3: Важность признаков...")
cat_cols = ["brand", "body_type", "fuel_type", "transmission", "drive", "condition"]
for col in cat_cols:
    if col in base.columns:
        base[col + "_code"] = LabelEncoder().fit_transform(base[col].fillna("unknown").astype(str))

features = ["price", "year", "mileage", "car_age"]
features += [c + "_code" for c in cat_cols if c in base.columns]
features = [f for f in features if f in base.columns]

X = base[features].fillna(0)
y = base["total_contacts"]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

imp_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df.head(8), x="importance", y="feature")
plt.title("Важность факторов ликвидности")
plt.tight_layout()
plt.savefig("03_importance.png", dpi=150)
plt.show()
print("   Сохранено: 03_importance.png")

# ВИЗУАЛИЗАЦИЯ 4: Кривые ликвидности
print("\n[7/8] Визуализация 4: Кривые ликвидности...")
main_segments = ["Массовые", "Премиум", "Возрастные >15 лет", "Дорогие >3 млн"]
plt.figure(figsize=(12, 7))

for seg in main_segments:
    if seg not in base["segment"].values:
        continue
    ids = base[base["segment"] == seg]["id"].values
    seg_cnt = cnt[cnt["id"].isin(ids)]
    if len(seg_cnt) < 50:
        continue
    daily = seg_cnt.groupby("day")["cnt_contacts"].mean().reset_index()
    smoothed = daily["cnt_contacts"].rolling(3, min_periods=1).mean()
    if smoothed.max() > 0:
        norm = smoothed / smoothed.max()
        plt.plot(daily["day"], norm, label=seg, linewidth=2)

plt.xlabel("День жизни объявления")
plt.ylabel("Нормированные контакты (пик = 1)")
plt.title("Кривые ликвидности по сегментам")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_curves.png", dpi=150)
plt.show()
print("   Сохранено: 04_curves.png")

# ВИЗУАЛИЗАЦИЯ 5: Круговая диаграмма (5-я обязательная)
print("\n[8/8] Визуализация 5: Распределение сегментов...")
plt.figure(figsize=(10, 8))
seg_sizes = base['segment'].value_counts()
plt.pie(seg_sizes.values, labels=seg_sizes.index, autopct='%1.1f%%', startangle=90)
plt.title('Распределение объявлений по сегментам', fontsize=14)
plt.tight_layout()
plt.savefig("05_pie_chart.png", dpi=150)
plt.show()
print("   Сохранено: 05_pie_chart.png")

# Рекомендации по VAS
print("\n" + "="*70)
print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ ПО VAS")
print("="*70)

for seg in main_segments:
    if seg not in base["segment"].values:
        continue
    ids = base[base["segment"] == seg]["id"].values
    seg_cnt = cnt[cnt["id"].isin(ids)]
    if len(seg_cnt) < 50:
        continue
    daily = seg_cnt.groupby("day")["cnt_contacts"].mean().reset_index()
    smoothed = daily["cnt_contacts"].rolling(3, min_periods=1).mean()
    max_val = smoothed.max()
    vas_day = None
    for i, val in enumerate((smoothed / max_val).values):
        if val < 0.5 and i > 0:
            vas_day = daily["day"].iloc[i]
            break
    avg_contacts = base[base["segment"] == seg]["total_contacts"].mean()
    print(f"\n📌 {seg}")
    print(f"   Среднее контактов: {avg_contacts:.1f}")
    if vas_day:
        print(f"   ✅ Рекомендуемый день VAS: {vas_day}-й день")
        ex = base[base["segment"] == seg].iloc[0]
        print(f"   📌 Пример: {ex['brand']} {ex['model']}, {ex['year']} г., {ex['region']}")
    else:
        print(f"   ⚠️ VAS не рекомендуется")

base.to_csv("liquidity_result.csv", index=False)

print("\n" + "="*70)
print("✅ АНАЛИЗ ЗАВЕРШЁН!")
print("="*70)
print("\n📁 СОХРАНЁННЫЕ ФАЙЛЫ (5 визуализаций):")
print("   • 01_correlation.png — корреляционная матрица")
print("   • 02_segments.png — boxplot сегментов")
print("   • 03_importance.png — важность признаков")
print("   • 04_curves.png — кривые ликвидности")
print("   • 05_pie_chart.png — круговая диаграмма")
print("   • liquidity_result.csv — данные с сегментами")
print("="*70)
input("\nНажми Enter для выхода...")
