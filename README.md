# 📊 Kalp Krizi Tahmin Uygulaması

Bu proje, yapay zeka destekli bir kalp krizi risk tahmin sistemidir. Kullanıcılar ister **tekli hasta verisi** girerek bireysel tahmin alabilir, isterlerse **toplu CSV dosyası** yükleyerek toplu analiz ve raporlama yapabilir.

## 🧰 Teknolojiler
- Python
- Streamlit
- scikit-learn
- pandas
- matplotlib
- joblib

## 📆 Özellikler
- Tekli hasta verisi girişi ve risk tahmini
- Toplu veri yükleme (CSV) ve analiz
- Risk gruplarına göre pasta grafiği
- ROC Eğrisi ve AUC değeri hesaplama
- PDF çıktısı alma (grafikler dahil)

## 🔧 Nasıl çalıştırılır?
1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
2. Uygulamayı başlatın:
```bash
streamlit run kalp_super_ust_seviye.py
```

## 💾 Dosyalar
- `kalp_super_ust_seviye.py`: Ana uygulama dosyası
- `heart.csv`: Örnek veri seti
- `kalp_modeli.pkl`: Kaydedilmiş makine öğrenmesi modeli

## 📅 Proje Sahipleri
**Halil Can Aydın**  
**Çağla Çoban**  
Yapay Zeka Projesi | 2025

