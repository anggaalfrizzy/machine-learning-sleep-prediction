# ============================================================
# TUGAS 9 – KNN untuk Prediksi Kualitas Tidur
# Sesuai Modul Praktikum Kecerdasan Buatan BAB 9
# Dosen: Mohammad Bayu Anggara, S.Kom., M.Kom.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ─────────────────────────────────────────────────────────────
# LANGKAH 1 – MENGUMPULKAN DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  PREDIKSI KUALITAS TIDUR DENGAN K-NEAREST NEIGHBORS")
print("=" * 55)

df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
print(f"\n[1] Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print(df.head())

# ─────────────────────────────────────────────────────────────
# LANGKAH 2 – PREPROCESSING DATA
# ─────────────────────────────────────────────────────────────
print("\n[2] Preprocessing data...")

df = df.drop(columns=['Person ID'])

# Label Encoding untuk kolom kategorikal
le_gender     = LabelEncoder()
le_occupation = LabelEncoder()
le_bmi        = LabelEncoder()
le_disorder   = LabelEncoder()

df['Gender']         = le_gender.fit_transform(df['Gender'])
df['Occupation']     = le_occupation.fit_transform(df['Occupation'])
df['BMI Category']   = le_bmi.fit_transform(df['BMI Category'])
df['Sleep Disorder'] = le_disorder.fit_transform(df['Sleep Disorder'].fillna('None'))

# Pisah Blood Pressure → Systolic & Diastolic
df[['Systolic', 'Diastolic']] = (
    df['Blood Pressure'].str.split('/', expand=True).astype(int)
)
df = df.drop(columns=['Blood Pressure'])

# Target: kategorikan Quality of Sleep
# Rendah (1-5) | Sedang (6-7) | Tinggi (8-9)
def kategorikan_kualitas(skor):
    if skor <= 5:
        return 'Rendah'
    elif skor <= 7:
        return 'Sedang'
    else:
        return 'Tinggi'

df['Kualitas Tidur'] = df['Quality of Sleep'].apply(kategorikan_kualitas)
print(f"   Distribusi target:\n{df['Kualitas Tidur'].value_counts().to_string()}")

feature_cols = [
    'Age', 'Gender', 'Occupation', 'Sleep Duration',
    'Physical Activity Level', 'Stress Level',
    'BMI Category', 'Heart Rate', 'Daily Steps',
    'Systolic', 'Diastolic', 'Sleep Disorder'
]
quality_labels = ['Rendah', 'Sedang', 'Tinggi']

X = df[feature_cols]
y = df['Kualitas Tidur']

# ─────────────────────────────────────────────────────────────
# LANGKAH 3 – SPLIT & NORMALISASI DATA
# ─────────────────────────────────────────────────────────────
print("\n[3] Split data (80% train, 20% test) & normalisasi...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_all_sc   = scaler.transform(X)

print(f"   Data train: {len(X_train)} | Data test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# LANGKAH 4 – NearestNeighbors (sesuai modul 9.6)
# ─────────────────────────────────────────────────────────────
print("\n[4] Membangun model NearestNeighbors (cosine similarity)...")

# Model NearestNeighbors – persis seperti contoh di modul BAB 9.6
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(X_train_sc)

# Cari 3 tetangga terdekat untuk data uji pertama
distances, indices = nn_model.kneighbors(X_test_sc[0:1], n_neighbors=3)
print(f"   Contoh 3 tetangga terdekat (indeks) : {indices[0]}")
print(f"   Jarak cosine                        : {distances[0].round(4)}")

# Prediksi manual dengan voting mayoritas (sesuai modul 9.2 – Menentukan prediksi)
def voting_knn(nn_model, y_train_arr, X_new, k=3):
    """Prediksi kelas dengan voting mayoritas dari k tetangga terdekat."""
    dists, idxs = nn_model.kneighbors(X_new, n_neighbors=k)
    predictions = []
    for idx_list in idxs:
        neighbor_labels = y_train_arr.iloc[idx_list].values
        unique, counts  = np.unique(neighbor_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])
    return np.array(predictions)

# Demo voting untuk data test
y_pred_demo = voting_knn(nn_model, y_train, X_test_sc, k=3)
print(f"   Akurasi demo voting (k=3): {accuracy_score(y_test, y_pred_demo):.4f}")

# ─────────────────────────────────────────────────────────────
# LANGKAH 5 – ELBOW METHOD (sesuai modul 9.4)
# ─────────────────────────────────────────────────────────────
print("\n[5] Elbow Method – mencari nilai K optimal...")

error_rates = []
k_range     = range(1, 21)

for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_tmp.fit(X_train_sc, y_train)
    error_rates.append(1 - accuracy_score(y_test, knn_tmp.predict(X_test_sc)))

# Pilih k terbaik via Cross-Validation (sesuai modul 9.4)
cv_per_k = []
for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_tmp.fit(X_train_sc, y_train)
    cv_per_k.append(cross_val_score(knn_tmp, X_all_sc, y, cv=5).mean())

best_k = list(k_range)[cv_per_k.index(max(cv_per_k))]
# Jika k=1 memiliki error 0 (overfitting), ambil k dari cv terbaik yang > 1
if best_k == 1:
    cv_no_k1 = cv_per_k[1:]
    best_k   = list(k_range)[1:][cv_no_k1.index(max(cv_no_k1))]

print(f"   K optimal (CV terbaik) : {best_k}")

# Plot Elbow
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), error_rates, marker='o', color='royalblue',
        linewidth=2, markersize=7, label='Error Rate')
ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.8,
           label=f'K Optimal = {best_k}')
ax.set_xlabel('Nilai K', fontsize=13)
ax.set_ylabel('Error Rate', fontsize=13)
ax.set_title('Elbow Method – Pemilihan Nilai K Optimal', fontsize=15)
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xticks(list(k_range))
fig.tight_layout()
fig.savefig('static/elbow_plot.png', dpi=100)
plt.close()
print("   Elbow plot disimpan → static/elbow_plot.png")

# ─────────────────────────────────────────────────────────────
# LANGKAH 6 – TRAINING MODEL FINAL
# ─────────────────────────────────────────────────────────────
print(f"\n[6] Training model final KNeighborsClassifier (k={best_k})...")

knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
knn.fit(X_train_sc, y_train)

y_pred = knn.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)
print(f"   Akurasi Test Set : {acc:.4f} ({acc:.2%})")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=quality_labels,
                             zero_division=0))

# ─────────────────────────────────────────────────────────────
# LANGKAH 7 – CROSS-VALIDATION (sesuai modul 9.4)
# ─────────────────────────────────────────────────────────────
print("[7] Cross-Validation (5-fold)...")

cv_scores = cross_val_score(knn, X_all_sc, y, cv=5)
print(f"   Skor tiap fold : {cv_scores.round(4)}")
print(f"   Rata-rata CV   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────
# LANGKAH 8 – VISUALISASI
# ─────────────────────────────────────────────────────────────
print("\n[8] Membuat visualisasi...")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=quality_labels)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=quality_labels, yticklabels=quality_labels,
            linewidths=0.5)
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title(f'Confusion Matrix – KNN (k={best_k})', fontsize=14)
fig.tight_layout()
fig.savefig('static/confusion_matrix.png', dpi=100)
plt.close()
print("   Confusion matrix → static/confusion_matrix.png")

# Distribusi Kualitas Tidur
color_map = {'Rendah': '#e74c3c', 'Sedang': '#f39c12', 'Tinggi': '#27ae60'}
counts    = y.value_counts().reindex(quality_labels)
fig, ax   = plt.subplots(figsize=(7, 4))
bars = ax.bar(counts.index, counts.values,
              color=[color_map[c] for c in counts.index],
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha='center', fontsize=12, fontweight='bold')
ax.set_xlabel('Kategori Kualitas Tidur', fontsize=12)
ax.set_ylabel('Jumlah Data', fontsize=12)
ax.set_title('Distribusi Kualitas Tidur dalam Dataset', fontsize=14)
fig.tight_layout()
fig.savefig('static/distribution.png', dpi=100)
plt.close()
print("   Distribusi plot → static/distribution.png")

# ─────────────────────────────────────────────────────────────
# LANGKAH 9 – SIMPAN MODEL
# ─────────────────────────────────────────────────────────────
print("\n[9] Menyimpan model & artefak...")

model_data = {
    'knn'            : knn,
    'nn_model'       : nn_model,
    'scaler'         : scaler,
    'le_gender'      : le_gender,
    'le_occupation'  : le_occupation,
    'le_bmi'         : le_bmi,
    'le_disorder'    : le_disorder,
    'feature_cols'   : feature_cols,
    'best_k'         : best_k,
    'accuracy'       : acc,
    'cv_scores'      : cv_scores,
    'cv_mean'        : cv_scores.mean(),
    'cv_std'         : cv_scores.std(),
    'class_report'   : classification_report(y_test, y_pred, output_dict=True,
                                              zero_division=0),
    'quality_labels' : quality_labels,
    'occupations'    : list(le_occupation.classes_),
    'bmi_categories' : list(le_bmi.classes_),
    'sleep_disorders': list(le_disorder.classes_),
    'genders'        : list(le_gender.classes_),
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("   Model disimpan → model.pkl")
print("\n" + "=" * 55)
print("  RINGKASAN HASIL")
print("=" * 55)
print(f"  Dataset      : {len(df)} baris, {len(feature_cols)} fitur")
print(f"  Train / Test : {len(X_train)} / {len(X_test)}")
print(f"  K Optimal    : {best_k} (Cross-Validation)")
print(f"  Akurasi Test : {acc:.2%}")
print(f"  CV Accuracy  : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
print("=" * 55)
