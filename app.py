from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

knn          = model_data['knn']
scaler       = model_data['scaler']
le_gender    = model_data['le_gender']
le_occupation = model_data['le_occupation']
le_bmi       = model_data['le_bmi']
le_disorder  = model_data['le_disorder']
feature_cols = model_data['feature_cols']
best_k       = model_data['best_k']
accuracy     = model_data['accuracy']
cv_mean      = model_data['cv_mean']
cv_std       = model_data['cv_std']
class_report = model_data['class_report']
occupations  = model_data['occupations']
bmi_cats     = model_data['bmi_categories']
sleep_dis    = model_data['sleep_disorders']


@app.route('/')
def index():
    return render_template('index.html',
                           occupations=occupations,
                           bmi_cats=bmi_cats,
                           sleep_dis=sleep_dis)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        age          = int(request.form['age'])
        gender       = request.form['gender']
        occupation   = request.form['occupation']
        sleep_dur    = float(request.form['sleep_duration'])
        phys_act     = int(request.form['physical_activity'])
        stress       = int(request.form['stress_level'])
        bmi_cat      = request.form['bmi_category']
        heart_rate   = int(request.form['heart_rate'])
        daily_steps  = int(request.form['daily_steps'])
        blood_press  = request.form['blood_pressure']
        disorder     = request.form['sleep_disorder']

        systolic, diastolic = map(int, blood_press.split('/'))

        gender_enc    = le_gender.transform([gender])[0]
        occupation_enc = le_occupation.transform([occupation])[0]
        bmi_enc       = le_bmi.transform([bmi_cat])[0]
        disorder_enc  = le_disorder.transform([disorder])[0]

        features = np.array([[age, gender_enc, occupation_enc, sleep_dur,
                               phys_act, stress, bmi_enc, heart_rate,
                               daily_steps, systolic, diastolic, disorder_enc]])

        features_sc = scaler.transform(features)
        prediction  = knn.predict(features_sc)[0]
        proba       = knn.predict_proba(features_sc)[0]
        classes     = knn.classes_

        prob_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

        tips = {
            'Rendah': [
                'Tingkatkan durasi tidur menjadi 7-9 jam per malam.',
                'Kurangi tingkat stres dengan meditasi atau relaksasi.',
                'Hindari kafein 6 jam sebelum tidur.',
                'Konsultasikan dengan dokter jika ada gangguan tidur.',
            ],
            'Sedang': [
                'Pertahankan rutinitas tidur yang konsisten.',
                'Tingkatkan aktivitas fisik minimal 30 menit per hari.',
                'Jaga pola makan sehat dan hindari makanan berat malam hari.',
                'Ciptakan lingkungan tidur yang nyaman dan gelap.',
            ],
            'Tinggi': [
                'Kualitas tidur Anda sudah sangat baik, pertahankan!',
                'Tetap jaga aktivitas fisik dan pola makan sehat.',
                'Lakukan pemeriksaan kesehatan rutin.',
                'Bagikan kebiasaan baik Anda kepada orang sekitar.',
            ],
        }

        return render_template('result.html',
                               prediction=prediction,
                               prob_dict=prob_dict,
                               tips=tips[prediction],
                               input_data={
                                   'Usia': age,
                                   'Jenis Kelamin': gender,
                                   'Pekerjaan': occupation,
                                   'Durasi Tidur': f'{sleep_dur} jam',
                                   'Aktivitas Fisik': f'{phys_act} menit/hari',
                                   'Tingkat Stres': f'{stress}/10',
                                   'Kategori BMI': bmi_cat,
                                   'Detak Jantung': f'{heart_rate} bpm',
                                   'Langkah Harian': f'{daily_steps:,}',
                                   'Tekanan Darah': blood_press,
                                   'Gangguan Tidur': disorder,
                               })

    except Exception as e:
        return render_template('index.html',
                               error=str(e),
                               occupations=occupations,
                               bmi_cats=bmi_cats,
                               sleep_dis=sleep_dis)


@app.route('/dashboard')
def dashboard():
    report = class_report
    metrics = {
        'accuracy': round(accuracy * 100, 2),
        'cv_mean': round(cv_mean * 100, 2),
        'cv_std': round(cv_std * 100, 2),
        'best_k': best_k,
        'rendah': report.get('Rendah', {}),
        'sedang': report.get('Sedang', {}),
        'tinggi': report.get('Tinggi', {}),
    }
    return render_template('dashboard.html', metrics=metrics)


@app.route('/about')
def about():
    return render_template('about.html')


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
