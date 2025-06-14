# interpolation.py
from base_module import BaseModule
import streamlit as st
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class InterpolationModule(BaseModule):
    def __init__(self):
        super().__init__()

    def run(self):
        st.header("İnterpolasyon Teknikleri")
        st.markdown("""
        Bu modül, veri noktaları arasında sürekli eğriler oluşturur:
        - **Doğrusal İnterpolasyon**: Veri noktalarını düz çizgilerle birleştirir. Basit, ancak keskin köşeler üretir (\( O(n) \)).
        - **Kübik Spline**: Pürüzsüz eğriler oluşturur, her segmentte kübik polinom kullanır (\( O(n) \)).
        **Uygulama**: Sensör verisi yumuşatma, eğri uydurma.
        **Giriş Formatı**: X ve Y noktaları virgülle ayrılmış sayılar (örn: '0, 1, 2', '0, 0.8, 0.9').
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Veri Girişi")
            st.markdown("""
            **Talimatlar**:
            - **X Noktaları**: Artan sırayla sayılar (örn: '0, 1, 2, 3').
            - **Y Noktaları**: X ile aynı sayıda değer (örn: '0, 0.8, 0.9, 0.1').
            - Kübik Spline için en az 4 nokta önerilir.
            """)
            x_str = st.text_input("X Noktaları:", "0, 1, 2, 3, 4, 5", key="interp_x")
            y_str = st.text_input("Y Noktaları:", "0, 0.8, 0.9, 0.1, -0.8, -1.0", key="interp_y")

            try:
                x_data = np.array([float(x.strip()) for x in x_str.split(',')])
                y_data = np.array([float(y.strip()) for y in y_str.split(',')])
                if len(x_data) != len(y_data):
                    st.warning("X ve Y veri noktası sayıları eşit olmalıdır!")
                    return
                if len(x_data) < 2:
                    st.warning("İnterpolasyon için en az 2 veri noktası gereklidir.")
                    return
                if np.any(np.diff(x_data) <= 0):
                    st.warning("X noktaları artan sırayla olmalıdır!")
                    return
            except ValueError:
                st.warning("Geçerli sayısal X ve Y değerleri girin.")
                return

        x_dense = np.linspace(min(x_data), max(x_data), 300)
        y_linear = np.interp(x_dense, x_data, y_data)
        y_spline = None
        if len(x_data) >= 4:
            cs = CubicSpline(x_data, y_data)
            y_spline = cs(x_dense)
        else:
            st.warning("Kübik Spline için en az 4 nokta önerilir.")

        with col2:
            st.subheader("İnterpolasyon Grafiği")
            st.markdown("""
            **Grafik Yorumlama**:
            - Mavi noktalar: Girilen veri noktaları.
            - Kırmızı çizgi: Doğrusal interpolasyon (segmentler arası düz çizgiler).
            - Yeşil çizgi: Kübik Spline (pürüzsüz eğri, 4+ nokta için).
            """)
            plots = [y_linear]
            labels = ["Doğrusal İnterpolasyon"]
            if y_spline is not None:
                plots.append(y_spline)
                labels.append("Kübik Spline İnterpolasyonu")
            self.plot_graph(x_dense, plots, xlabel="X", ylabel="Y", title="İnterpolasyon",
                labels=labels, points=[(x_data[i], y_data[i], "") for i in range(len(x_data))],
                figsize=(4, 3))

            # Animasyon
            if st.button("İnterpolasyon Animasyonunu Göster", key="interp_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, interpolasyon sürecini adım adım gösterir:
                - **Doğrusal İnterpolasyon**: Veri noktaları arasında düz çizgiler sırayla çizilir.
                - **Kübik Spline**: Noktalara uyan pürüzsüz eğri, noktalar eklendikçe güncellenir.
                - Mavi noktalar veri noktalarını, kırmızı/yeşil çizgiler interpolasyon eğrilerini gösterir.
                - Her adımda, eklenen nokta veya segment belirtilir.
                """)
                try:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.scatter(x_data, y_data, color='blue', label="Veri Noktaları")
                    line_linear, = ax.plot([], [], 'r-', label="Doğrusal İnterpolasyon")
                    line_spline, = ax.plot([], [], 'g-', label="Kübik Spline" if len(x_data) >= 4 else "")
                    desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                    ax.set_xlabel("X", fontsize=8)
                    ax.set_ylabel("Y", fontsize=8)
                    ax.set_title("İnterpolasyon Süreci", fontsize=10)
                    ax.legend(fontsize=8)
                    fig.subplots_adjust(bottom=0.25)

                    def animate(i):
                        if i < len(x_data) - 1:
                            # Doğrusal interpolasyon
                            x_segment = np.linspace(x_data[i], x_data[i+1], 50)
                            y_segment = np.interp(x_segment, x_data[:i+2], y_data[:i+2])
                            line_linear.set_data(x_segment, y_segment)
                            desc_text.set_text(f"Doğrusal: Segment {i+1} ({x_data[i]:.1f}, {x_data[i+1]:.1f})")
                            if len(x_data) >= 4 and i >= 3:
                                # Kübik Spline (4 nokta tamamlandığında)
                                cs = CubicSpline(x_data[:i+2], y_data[:i+2])
                                y_spline_partial = cs(x_dense)
                                line_spline.set_data(x_dense, y_spline_partial)
                                desc_text.set_text(f"Spline: {i+2} nokta eklendi")
                        return [line_linear, line_spline, desc_text]

                    self.save_animation(fig, animate, frames=len(x_data)-1, interval=500)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Animasyon hatası: {e}")

        self.sensor_smoothing()

    def sensor_smoothing(self):
        st.subheader("Uygulama: Gürültülü Sensör Verisi Yumuşatma")
        st.markdown("""
        Bu uygulama, gürültülü sensör verisini interpolasyonla yumuşatır:
        - **Veri**: Gerçek sinyal (\( \sin(x) + 0.5\cos(0.5x) \)) üzerine gürültü eklenir.
        - **Doğrusal İnterpolasyon**: Gürültülü noktalar arasında düz çizgiler çizer.
        - **Kübik Spline**: Pürüzsüz bir eğriyle veriyi yumuşatır (4+ nokta için).
        - **Amaç**: Gürültüyü azaltarak gerçek sinyale yaklaşmak.
        """)
        num_points = st.slider("Veri Noktası Sayısı:", 5, 50, 10)
        noise_level = st.slider("Gürültü Seviyesi:", 0.0, 1.0, 0.2)

        x_orig = np.linspace(0, 10, num_points)
        y_true = np.sin(x_orig) + 0.5 * np.cos(0.5 * x_orig)
        y_noisy = y_true + noise_level * np.random.randn(num_points)

        x_dense = np.linspace(min(x_orig), max(x_orig), 200)
        y_linear_interp = np.interp(x_dense, x_orig, y_noisy)
        y_spline_interp = None
        if len(x_orig) >= 4:
            cs = CubicSpline(x_orig, y_noisy)
            y_spline_interp = cs(x_dense)

        st.subheader("Sensör Verisi Grafiği")
        plots = [y_linear_interp]
        labels = ["Doğrusal İnterpole Edilmiş"]
        if y_spline_interp is not None:
            plots.append(y_spline_interp)
            labels.append("Kübik Spline İnterpole Edilmiş")
        plots.append(np.sin(x_dense) + 0.5 * np.cos(0.5 * x_dense))
        labels.append("Gerçek Sinyal")
        self.plot_graph(x_dense, plots, xlabel="Zaman", ylabel="Sensör Değeri",
                       title="Sensör Verisi Yumuşatma", labels=labels,
                       points=[(x_orig[i], y_noisy[i], "") for i in range(len(x_orig))],
                       figsize=(4, 3))

        # Sensör animasyonu
        if st.button("Sensör Yumuşatma Animasyonunu Göster", key="sensor_anim"):
            st.markdown("""
            **Animasyon Açıklaması**:
            Bu animasyon, gürültülü sensör verisinin interpolasyonla yumuşatılmasını gösterir:
            - Gürültülü veri noktaları sırayla eklenir.
            - Doğrusal interpolasyon, noktalar arasında düz çizgiler çizer.
            - Kübik Spline (4+ nokta için), pürüzsüz bir eğri oluşturur.
            - Gerçek sinyal, karşılaştırma için gösterilir.
            """)
            try:
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot(x_dense, np.sin(x_dense) + 0.5 * np.cos(0.5 * x_dense), 'k--', label="Gerçek Sinyal")
                ax.scatter([], [], color='blue', label="Gürültülü Veri")
                line_linear, = ax.plot([], [], 'r-', label="Doğrusal İnterpolasyon")
                line_spline, = ax.plot([], [], 'g-', label="Kübik Spline" if len(x_orig) >= 4 else "")
                desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                ax.set_xlabel("Zaman", fontsize=8)
                ax.set_ylabel("Sensör Değeri", fontsize=8)
                ax.set_title("Sensör Yumuşatma", fontsize=10)
                ax.legend(fontsize=8)
                fig.subplots_adjust(bottom=0.25)

                def animate(i):
                    if i < len(x_orig) - 1:
                        # Gürültülü noktaları ekle
                        ax.scatter(x_orig[:i+2], y_noisy[:i+2], color='blue')
                        # Doğrusal interpolasyon
                        x_segment = np.linspace(x_orig[i], x_orig[i+1], 50)
                        y_segment = np.interp(x_segment, x_orig[:i+2], y_noisy[:i+2])
                        line_linear.set_data(x_segment, y_segment)
                        desc_text.set_text(f"Doğrusal: Nokta {i+2} eklendi")
                        if len(x_orig) >= 4 and i >= 3:
                            # Kübik Spline
                            cs = CubicSpline(x_orig[:i+2], y_noisy[:i+2])
                            y_spline_partial = cs(x_dense)
                            line_spline.set_data(x_dense, y_spline_partial)
                            desc_text.set_text(f"Spline: {i+2} nokta eklendi")
                    return [line_linear, line_spline, desc_text]

                self.save_animation(fig, animate, frames=len(x_orig)-1, interval=500)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Animasyon hatası: {e}")