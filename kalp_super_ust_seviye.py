import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from io import BytesIO
import os

st.set_page_config(page_title="Kalp Krizi Tahmin Raporu", layout="centered")
st.title("💓 Kalp Krizi Riski Tahmin Uygulaması")

st.markdown("""
Bu yapay zeka destekli uygulamada, **tekli** veya **toplu** hasta verileriyle kalp krizi riski tahmini yapabilir,
tahmin sonuçlarını görselleştirebilir ve **rapor halinde indirebilirsiniz**.
""")

model = joblib.load("kalp_modeli.pkl")
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
scaler = StandardScaler()
scaler.fit(X)

toplu_veri = None

st.sidebar.header("🔍 Veri Girişi")
secilen_mod = st.sidebar.radio("Mod Seçimi:", ("Tekli Hasta Girişi", "Toplu CSV Yükle"))

if secilen_mod == "Tekli Hasta Girişi":
    age = st.sidebar.slider("Yaş", 20, 100, 50)
    sex = st.sidebar.selectbox("Cinsiyet", [("Erkek", 1), ("Kadın", 0)])
    cp = st.sidebar.selectbox("Göğüs Ağrısı Tipi", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("İstirahat Kan Basıncı", 80, 200, 120)
    chol = st.sidebar.slider("Kolesterol", 100, 600, 240)
    fbs = st.sidebar.selectbox("Açlık Kan Şekeri > 120", [("Evet", 1), ("Hayır", 0)])
    restecg = st.sidebar.selectbox("EKG Sonucu", [0, 1, 2])
    thalach = st.sidebar.slider("Maksimum Kalp Atış Hızı", 60, 220, 150)
    exang = st.sidebar.selectbox("Egzersize Bağlı Angina", [("Evet", 1), ("Hayır", 0)])
    oldpeak = st.sidebar.slider("ST Depresyonu", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("ST Segment Eğimi", [0, 1, 2])
    ca = st.sidebar.slider("Boyalı Damar Sayısı", 0, 3, 0)
    thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

    veri = pd.DataFrame([{ "age": age, "sex": sex[1], "cp": cp, "trestbps": trestbps, "chol": chol,
                           "fbs": fbs[1], "restecg": restecg, "thalach": thalach, "exang": exang[1],
                           "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal }])

    if st.button("🔮 Tahmin Et"):
        veri_scaled = scaler.transform(veri)
        tahmin = model.predict(veri_scaled)
        proba = model.predict_proba(veri_scaled)[0][1]

        if tahmin[0] == 1:
            st.error(f"⚠️ Kalp krizi riski VAR! Olasılık: %{proba * 100:.2f}")
        else:
            st.success(f"✅ Kalp krizi riski YOK. Olasılık: %{proba * 100:.2f}")

else:
    uploaded_file = st.sidebar.file_uploader("📂 Hasta Verisi Yükleyin (CSV)", type=["csv"])
    if uploaded_file:
        toplu_veri = pd.read_csv(uploaded_file)
        veri_scaled = scaler.transform(toplu_veri)
        tahminler = model.predict(veri_scaled)
        olasiliklar = model.predict_proba(veri_scaled)[:, 1]

        toplu_veri["Kalp Krizi Riski"] = tahminler
        toplu_veri["Risk Olasılığı (%)"] = olasiliklar * 100

        def etiketle(risk):
            if risk >= 70:
                return "Yüksek Risk"
            elif risk >= 40:
                return "Orta Risk"
            else:
                return "Düşük Risk"

        toplu_veri["Risk Grubu"] = toplu_veri["Risk Olasılığı (%)"].apply(etiketle)

        st.subheader("📋 Toplu Tahmin Sonuçları")
        st.dataframe(toplu_veri)

        st.subheader("📈 Rapor Özeti")
        total = len(toplu_veri)
        riskli = (toplu_veri["Kalp Krizi Riski"] == 1).sum()
        risk_oran = riskli / total * 100
        ort_risk = toplu_veri["Risk Olasılığı (%)"].mean()

        st.markdown(f"""
        - 👥 Toplam hasta sayısı: **{total}**
        - ❗ Riskli hasta sayısı: **{riskli}**
        - 📊 Ortalama risk olasılığı: **%{ort_risk:.2f}**
        - 🔥 Riskli hasta oranı: **%{risk_oran:.2f}**
        """)

        st.subheader("🍰 Risk Gruplarına Göre Dağılım")
        grup_sayilari = toplu_veri["Risk Grubu"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(grup_sayilari, labels=grup_sayilari.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        st.subheader("📉 ROC Eğrisi (Receiver Operating Characteristic)")
        fpr, tpr, _ = roc_curve(df["target"], model.predict_proba(X)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"ROC Eğrisi (AUC = {roc_auc:.2f})")
        ax2.plot([0, 1], [0, 1], "--", color="navy")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Eğrisi")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

        # İndirme
        st.download_button("📥 Raporu İndir (CSV)", data=toplu_veri.to_csv(index=False).encode("utf-8"),
                           file_name="kalp_krizi_raporu.csv", mime="text/csv")

        if st.button("📄 PDF Raporu Oluştur"):
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            width, height = A4
            c.setTitle("Kalp Krizi Tahmin Raporu")

            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(width / 2, height - 50, "Kalp Krizi Tahmin Raporu")

            y = height - 80
            c.setFont("Helvetica", 12)
            c.drawString(50, y, f"Toplam Hasta Sayısı: {total}")
            y -= 20
            c.drawString(50, y, f"Riskli Hasta Sayısı: {riskli}")
            y -= 20
            c.drawString(50, y, f"Ortalama Risk Olasılığı: %{ort_risk:.2f}")
            y -= 20
            c.drawString(50, y, f"Riskli Hasta Oranı: %{risk_oran:.2f}")

            pie_path = "pie_chart.png"
            fig1.savefig(pie_path, bbox_inches="tight")
            c.drawImage(pie_path, 50, y - 300, width=10 * cm, preserveAspectRatio=True)
            os.remove(pie_path)

            roc_path = "roc_curve.png"
            fig2.savefig(roc_path, bbox_inches="tight")
            c.drawImage(roc_path, 50, y - 600, width=10 * cm, preserveAspectRatio=True)
            os.remove(roc_path)

            c.setFont("Helvetica-Oblique", 10)
            c.drawString(50, 40, "🧠 Bu rapor yapay zeka destekli bir modelle Streamlit kullanılarak oluşturulmuştur. © 2025")

            c.save()
            buffer.seek(0)

            st.download_button("📥 PDF Raporu İndir", data=buffer,
                               file_name="kalp_krizi_tahmin_raporu.pdf",
                               mime="application/pdf")

st.markdown("---")
st.caption("📍 Hazırlayan: Halil Can Aydın ve Çağla Çoban | Yapay Zeka Projesi - 2025")
