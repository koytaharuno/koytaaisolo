# 🎓 Academic Performance Prediction
**Supervised Machine Learning Project**

Project ini merupakan implementasi **Artificial Intelligence (AI)** menggunakan pendekatan **Supervised Machine Learning** untuk memprediksi **kategori performa akademik mahasiswa** berdasarkan data akademik.

---

## 📌 Gambaran Umum

Sistem ini memprediksi **Academic Performance** ke dalam tiga kategori:

- **Low**
- **Medium**
- **High**

Prediksi dilakukan menggunakan algoritma **Random Forest Classifier** dengan data akademik sebagai input.

> ⚠️ Catatan penting:
> Project ini **bukan sistem berbasis aturan**, melainkan model AI yang **belajar pola dari data berlabel**.

---

## 🎯 Tujuan Project

- Menerapkan **Supervised Learning**
- Menyelesaikan permasalahan **klasifikasi**
- Menggunakan dataset dengan **≥ 1000 data**
- Mengimplementasikan algoritma **Machine Learning**
- Menyediakan sistem prediksi berbasis data baru (user input)

---

## 🧠 Konsep Dasar Sistem

Alur kerja sistem AI ini adalah:

1. Memuat dataset akademik  
2. Membersihkan data (data cleaning)  
3. Memilih fitur akademik yang relevan  
4. Membentuk label performa akademik  
5. Melatih model AI  
6. Mengevaluasi performa model  
7. Menggunakan model untuk prediksi data baru  

---

## 📂 Struktur Dataset

Setiap baris data merepresentasikan **satu mahasiswa** dengan fitur berikut:

| Fitur | Deskripsi |
|---|---|
| `FinalGrade` | Nilai akhir mahasiswa |
| `Attendance (%)` | Persentase kehadiran |
| `StudyHoursPerWeek` | Jam belajar per minggu |
| `PreviousGrade` | Nilai akademik sebelumnya |

---

## 📦 Import Library

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
```

---

## 📂 Load Dataset

```python
df = pd.read_csv("data.csv")
print("Jumlah data:", len(df))
```

---

## 🧹 Data Cleaning

```python
required_features = [
    "FinalGrade",
    "Attendance (%)",
    "StudyHoursPerWeek",
    "PreviousGrade"
]

df = df.dropna(subset=required_features).copy()
```

---

## 🎛️ Feature Selection

```python
X = df[required_features]
```

---

## 🧮 Performance Score

```python
df["performance_score"] = (
    0.35 * df["FinalGrade"] +
    0.25 * df["Attendance (%)"] +
    0.25 * df["StudyHoursPerWeek"] +
    0.15 * df["PreviousGrade"]
)
```

---

## 🏷️ Labeling

```python
low = df["performance_score"].quantile(0.33)
high = df["performance_score"].quantile(0.66)

def label_performance(score):
    if score <= low:
        return "Low"
    elif score <= high:
        return "Medium"
    else:
        return "High"

df["performance_label"] = df["performance_score"].apply(label_performance)
```

---

## 🔢 Label Encoding

```python
le = LabelEncoder()
y = le.fit_transform(df["performance_label"])
```

---

## ✂️ Train–Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 🤖 Training Model

```python
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
```

---

## 📊 Evaluation

```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🧪 Prediction

Model dapat digunakan untuk memprediksi performa mahasiswa baru berdasarkan input user.

---

## ✅ Kesimpulan

- Supervised Machine Learning berhasil diterapkan
- Model mempelajari pola akademik dari data
- Random Forest cocok untuk data tabular
- Sistem siap dikembangkan lebih lanjut

---

## 📝 Catatan Akademik

Label performa bersifat **proxy** untuk keperluan supervised learning.
Model AI tidak mengetahui cara label dibuat, hanya mempelajari pola dari data berlabel.
