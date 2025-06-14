# numerical_integration.py
import streamlit as st
import numpy as np
from scipy import integrate
from base_module import BaseModule
import matplotlib.pyplot as plt
import sympy as sp

class NumericalIntegrationModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'integration_results' not in self.session_state:
            self.session_state.integration_results = []

    def run(self):
        st.header("Sayısal İntegrasyon Yöntemleri")
        st.markdown("""
        Bu modül, belirli integralleri sayısal yöntemlerle hesaplar. İki yöntem sunulmaktadır:
        - **Yamuk Kuralı**: Fonksiyonu yamuklarla yaklaşık olarak entegre eder. Basit ve ayrık veriler için uygundur.
        - **Adaptif Quad**: Fonksiyonu yüksek hassasiyetle entegre eder, ancak yalnızca tanımlı fonksiyonlar için çalışır.
        Sayısal integrasyon, analitik çözümü olmayan integraller veya sensör gibi ayrık veriler için kullanılır.
        """)

        # Teori Bölümü
        with st.expander("Teori: Sayısal İntegrasyon"):
            st.markdown("""
            Sayısal integrasyon, bir fonksiyonun belirli integralini yaklaşık olarak hesaplar:
            \[
            \int_a^b f(x) \, dx
            \]
            Analitik çözümü zor veya imkansız olan durumlarda kullanılır.

            ### Yamuk Kuralı
            - **Fikir**: Entegrasyon aralığını \( n \) eşit parçaya böler ve her parçayı bir yamukla yaklaşıklar.
            - **Formül**:
              \[
              \int_a^b f(x) \, dx \approx \frac{h}{2} \left[ f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n) \right], \quad h = \frac{b-a}{n}
              \]
            - **Hata**: \( O(h^2) \), yani adım boyu \( h \) küçüldükçe hata karesel azalır.
            - **Avantaj**: Basit, ayrık veriler için uygun.
            - **Dezavantaj**: Daha az hassas.

            ### Adaptif Quad
            - **Fikir**: Fonksiyonu adaptif olarak değerlendirir, hata tahminine göre adım boyutunu ayarlar.
            - **Hata**: Çok küçük (genellikle \( 10^{-10} \) seviyesinde).
            - **Avantaj**: Yüksek hassasiyet.
            - **Dezavantaj**: Yalnızca tanımlı fonksiyonlar için çalışır, ayrık veriyle kullanılamaz.

            ### Uygulamalar
            - **Fizik**: Mesafe, iş, enerji hesaplama.
            - **İstatistik**: Olasılık yoğunluk fonksiyonlarının integrali.
            - **Mühendislik**: Ortalama değer, alan hesaplama.
            """)
            st.latex(r"\int_a^b f(x) \, dx \approx \frac{h}{2} \left[ f(x_0) + 2 \sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fonksiyon ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Fonksiyon**: Matematiksel bir fonksiyon girin (örneğin, `sin(x)`, `x**2`). Desteklenen işlemler: `sin`, `cos`, `exp`, `sqrt`, `log`, `tan`.
            - **Entegrasyon Aralığı**: İntegralin hesaplanacağı \( [a, b] \) aralığını belirleyin.
            - **Bölme Sayısı (n)**: Yamuk kuralı için aralığın kaç parçaya bölüneceğini seçin (daha fazla \( n \), daha doğru sonuç).
            - **Ayrık Veri**: Sensör verisi gibi x ve y noktalarını manuel olarak girebilirsiniz.
            """)

            func_str = st.text_input("Fonksiyon f(x):", "sin(x)", key="integ_func_str")
            example_funcs = ["sin(x)", "x**2 + 2*x + 1", "exp(-x)", "1/(1+x**2)"]
            selected_example = st.selectbox("Örnek Fonksiyon:", [""] + example_funcs, key="integ_example_select")
            if selected_example:
                func_str = selected_example

            # Fonksiyonu ayrıştır
            f, expr = self.parse_function_string(func_str)
            if expr:
                st.latex(f"f(x) = {sp.latex(expr)}")
            else:
                st.error("Geçerli bir fonksiyon girin (örneğin, 'sin(x)' veya 'x**2').")
                return

            a = st.number_input("Entegrasyon Başlangıcı (a):", value=0.0, key="integ_a")
            b = st.number_input("Entegrasyon Sonu (b):", value=1.0, key="integ_b")
            n = st.number_input("Bölme Sayısı (n, yamuk kuralı için):", value=100, min_value=2, step=2, key="integ_n")
            use_data = st.checkbox("Ayrık Veri Kullan (örn: sensör verisi)", key="integ_use_data")

            x_data, y_data = None, None
            if use_data:
                st.markdown("""
                **Ayrık Veri Girişi**:
                - X ve Y noktalarını virgülle ayırarak girin.
                - Örnek: X: `0, 0.25, 0.5`, Y: `0, 0.247, 0.479`.
                - X ve Y noktası sayıları eşit olmalıdır.
                """)
                x_str = st.text_input("X Noktaları (virgülle ayrılmış):", "0, 0.25, 0.5, 0.75, 1.0", key="integ_x_data")
                y_str = st.text_input("Y Noktaları (virgülle ayrılmış):", "0, 0.247, 0.479, 0.681, 0.841", key="integ_y_data")
                try:
                    x_data = np.array([float(x.strip()) for x in x_str.split(',')])
                    y_data = np.array([float(y.strip()) for x in y_str.split(',')])
                    if len(x_data) != len(y_data):
                        st.error("X ve Y veri noktası sayıları eşit olmalıdır!")
                        return
                    if len(x_data) < 2:
                        st.error("En az 2 veri noktası gereklidir.")
                        return
                    if not np.all(np.diff(x_data) > 0):
                        st.error("X noktaları artan sırayla olmalıdır!")
                        return
                except ValueError:
                    st.error("Geçerli sayısal X ve Y değerleri girin.")
                    return

        # Entegrasyon hesaplamaları
        results = []
        x_vals = np.linspace(a, b, n) if not use_data else x_data
        try:
            y_vals = f(x_vals) if not use_data else y_data
        except Exception as e:
            st.error(f"Fonksiyon değerlendirme hatası: {e}")
            return

        # Yamuk Kuralı
        try:
            trap_result = integrate.trapezoid(y_vals, x_vals)
            results.append({"Yöntem": "Yamuk Kuralı", "Sonuç": trap_result})
        except Exception as e:
            results.append({"Yöntem": "Yamuk Kuralı", "Hata": str(e)})

        # Quad (Adaptif Entegrasyon)
        if not use_data:
            try:
                quad_result, quad_error = integrate.quad(f, a, b)
                results.append({"Yöntem": "Quad", "Sonuç": quad_result, "Hata Tahmini": quad_error})
            except Exception as e:
                results.append({"Yöntem": "Quad", "Hata": str(e)})
        else:
            st.warning("Quad yöntemi ayrık veri için kullanılamaz.")

        # Analitik integral
        analytical_result = None
        if not use_data:
            try:
                x_sym = sp.symbols('x')
                expr_clean = func_str.replace('np.', '')  # np.sin -> sin
                integral_expr = sp.integrate(expr_clean, (x_sym, a, b))
                analytical_result = float(integral_expr)
                st.latex(f"\\int_{a}^{b} {sp.latex(expr)} \\, dx = {sp.latex(integral_expr)}")
                results.append({"Yöntem": "Analitik", "Sonuç": analytical_result})
            except Exception:
                st.warning("Analitik integral hesaplanamadı.")

        with col2:
            st.subheader("Sonuçlar ve Grafik")
            st.markdown("""
            **Grafik Yorumlama**:
            - Grafik, fonksiyonu \( f(x) \) ve entegrasyon aralığını \( [a, b] \) gösterir.
            - Entegral, grafikteki eğri altında kalan alanı temsil eder.
            - Yamuk kuralı, bu alanı yamuklarla yaklaşıklar; quad daha hassas bir tahmin sağlar.
            """)
            if results:
                st.dataframe(results)
            try:
                # Gölgelendirme ile entegrasyon alanını vurgulama
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(x_vals, y_vals, label="f(x)")
                ax.fill_between(x_vals, 0, y_vals, alpha=0.2, color='blue', label="Entegrasyon Alanı")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.set_title("Fonksiyon ve Entegrasyon Alanı")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Grafik çizme hatası: {e}")

            # Hata Analizi
            if analytical_result is not None:
                st.subheader("Hata Analizi")
                errors = [{"Yöntem": res["Yöntem"], "Hata": abs(res["Sonuç"] - analytical_result)}
                          for res in results if "Sonuç" in res and res["Yöntem"] != "Analitik"]
                if errors:
                    st.dataframe(errors)
                    st.markdown("""
                    **Hata Yorumlama**:
                    - **Yamuk Kuralı**: \( O(h^2) \) hata oranı, \( n \) arttıkça azalır.
                    - **Quad**: Çok düşük hata, genellikle analitik sonuca çok yakın.
                    """)

            # Animasyon
            if st.button("Yamuk Kuralı Animasyonunu Göster", key="trap_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, yamuk kuralının nasıl çalıştığını gösterir. Her adımda bir yamuk eklenir ve toplam alan yaklaşık olarak hesaplanır.
                """)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    y_max = np.max(y_vals) * 1.2 if np.max(y_vals) > 0 else 1.0
                    ax.set_xlim(x_vals[0], x_vals[-1])
                    ax.set_ylim(0, y_max)
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.set_title("Yamuk Kuralı Animasyonu")
                    ax.grid(True)
                    line, = ax.plot(x_vals, y_vals, label="f(x)")
                    ax.legend()

                    def animate(i, x_vals, y_vals, n):
                        ax.clear()
                        ax.plot(x_vals, y_vals, label="f(x)")
                        for j in range(min(i+1, n-1)):
                            ax.fill_between(x_vals[j:j+2], 0, y_vals[j:j+2], alpha=0.3, color='red')
                        ax.set_xlim(x_vals[0], x_vals[-1])
                        ax.set_ylim(0, y_max)
                        ax.set_xlabel("x")
                        ax.set_ylabel("f(x)")
                        ax.set_title(f"Yamuk Kuralı: {i+1} Yamuk")
                        ax.grid(True)
                        ax.legend()
                        return line,

                    self.save_animation(fig, lambda i: animate(i, x_vals, y_vals, n), frames=n-1, interval=200)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Animasyon hatası: {e}")

        self.distance_calculation_application(f, use_data, x_data, y_data)

    def distance_calculation_application(self, f, use_data, x_data, y_data):
        st.subheader("Uygulama: Hızdan Mesafe Hesaplama")
        st.markdown("""
        Hız fonksiyonu \( v(t) \) verildiğinde, mesafe şu şekilde hesaplanır:
        \[
        s(t) = \int_{t_0}^{t_{\text{end}}} v(t) \, dt
        \]
        **Gerçek Dünya Uygulaması**:
        - **Otomotiv**: Araç hız verilerinden kat edilen mesafe.
        - **Robotik**: Sensör verileriyle robot hareket analizi.
        - **Fizik**: Düzensiz hareketlerde mesafe hesaplama.
        """)
        st.latex(r"s(t) = \int_{t_0}^{t_{\text{end}}} v(t) \, dt")

        a_app = st.number_input("Başlangıç Zamanı (t0):", value=0.0, key="dist_a")
        b_app = st.number_input("Bitiş Zamanı (t_end):", value=1.0, key="dist_b")
        n_app = st.number_input("Bölme Sayısı (n):", value=100, min_value=2, step=2, key="dist_n")

        x_app = np.linspace(a_app, b_app, n_app) if not use_data else x_data
        try:
            y_app = f(x_app) if not use_data else y_data
        except Exception as e:
            st.error(f"Hız fonksiyonu değerlendirme hatası: {e}")
            return

        dist_results = []
        try:
            trap_dist = integrate.trapezoid(y_app, x_app)
            dist_results.append({"Yöntem": "Yamuk Kuralı", "Mesafe": trap_dist})
            if not use_data:
                quad_dist, quad_error = integrate.quad(f, a_app, b_app)
                dist_results.append({"Yöntem": "Quad", "Mesafe": quad_dist, "Hata Tahmini": quad_error})
        except Exception as e:
            st.error(f"Mesafe hesaplama hatası: {e}")
            return

        st.subheader("Mesafe Hesaplama Sonuçları")
        if dist_results:
            st.dataframe(dist_results)

        # Gürültü Ekleme
        add_noise = st.checkbox("Gürültü Ekle (Sensör Simülasyonu)", key="dist_noise")
        if add_noise:
            st.markdown("""
            **Gürültü Simülasyonu**:
            Gerçek dünya sensör verileri genellikle gürültü içerir. Gürültü ekleyerek, sayısal entegrasyonun bu koşullardaki performansını test edebilirsiniz.
            """)
            noise_level = st.slider("Gürültü Seviyesi:", 0.0, 0.5, 0.1, key="dist_noise_level")
            try:
                y_app_noisy = y_app + noise_level * np.random.randn(len(y_app))
                trap_dist_noisy = integrate.trapezoid(y_app_noisy, x_app)
                dist_results.append({"Yöntem": "Yamuk Kuralı (Gürültülü)", "Mesafe": trap_dist_noisy})
                st.dataframe(dist_results)
                self.plot_graph(x_app, [y_app, y_app_noisy], xlabel="Zaman (t)", ylabel="Hız v(t)",
                               title="Hız Fonksiyonu (Normal ve Gürültülü)",
                               labels=["Orijinal Hız", "Gürültülü Hız"], figsize=(6, 4))
            except Exception as e:
                st.error(f"Gürültülü hesaplama hatası: {e}")
        else:
            try:
                self.plot_graph(x_app, y_app, xlabel="Zaman (t)", ylabel="Hız v(t)",
                               title="Hız Fonksiyonu ve Mesafe",
                               labels=["v(t)"], figsize=(6, 4))
            except Exception as e:
                st.error(f"Grafik çizme hatası: {e}")

        # Quiz
        with st.expander("Quiz: Sayısal İntegrasyon"):
            st.markdown("Aşağıdaki soruyu yanıtlayın:")
            question = st.radio(
                "Yamuk kuralının hata oranı nedir?",
                ["O(h)", "O(h²)", "O(h³)", "O(1)"],
                key="quiz_integ"
            )
            if st.button("Cevabı Kontrol Et", key="quiz_integ_check"):
                if question == "O(h²)":
                    st.success("Doğru! Yamuk kuralının hata oranı \( O(h^2) \)'dir.")
                else:
                    st.error("Yanlış. Yamuk kuralının hata oranı \( O(h^2) \)'dir.")

        # Karşılaştırma Tablosu
        st.subheader("Yöntem Karşılaştırması")
        comparison = [
            {"Yöntem": "Yamuk Kuralı", "Hata Oranı": "O(h²)", "Uygunluk": "Ayrık veri, basit fonksiyonlar", "Hız": "Hızlı"},
            {"Yöntem": "Quad", "Hata Oranı": "Çok düşük (~10⁻¹⁰)", "Uygunluk": "Tanımlı fonksiyonlar", "Hız": "Daha yavaş"}
        ]
        st.dataframe(comparison)