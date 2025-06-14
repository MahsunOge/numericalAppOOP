# numerical_derivative.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from matplotlib import animation
from streamlit.components.v1 import html
from base_module import BaseModule

class NumericalDerivativeModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'derivative_results' not in self.session_state:
            self.session_state.derivative_results = []

    def forward_diff(self, f, x, h):
        """İleri fark türevi: f'(x) ≈ (f(x+h) - f(x))/h"""
        return (f(x + h) - f(x)) / h

    def backward_diff(self, f, x, h):
        """Geri fark türevi: f'(x) ≈ (f(x) - f(x-h))/h"""
        return (f(x) - f(x - h)) / h

    def central_diff(self, f, x, h):
        """Merkezi fark türevi: f'(x) ≈ (f(x+h) - f(x-h))/(2h)"""
        return (f(x + h) - f(x - h)) / (2 * h)

    def run(self):
        st.header("Sayısal Türev Yöntemleri")
        st.markdown("""
        Bu modül, bir fonksiyonun türevini sayısal yöntemlerle hesaplar. Üç yöntem sunulmaktadır:
        - **İleri Fark**: Gelecekteki bir noktayı kullanır, basit ama düşük hassasiyetlidir.
        - **Geri Fark**: Geçmişteki bir noktayı kullanır, ileri farka benzer.
        - **Merkezi Fark**: Simetrik noktalar kullanır, daha hassastır.
        """)

        # Teori Bölümü
        with st.expander("Teori: Sayısal Türev Yöntemleri"):
            st.markdown("""
            Sayısal türev, bir fonksiyonun analitik türevini hesaplamak yerine, yakın noktalardaki değer farklarını kullanarak türevi yaklaşık olarak hesaplar. Bu yöntemler, özellikle analitik türevin karmaşık olduğu durumlarda faydalıdır.

            ### Yöntemler ve Formüller
            1. **İleri Fark**:
               - **Formül**: 
               - **Hata**: O(h), yani h küçüldükçe hata doğrusal azalır.
               - **Avantaj**: Basit, hızlı hesaplama.
               - **Dezavantaj**: Düşük hassasiyet.

            2. **Geri Fark**:
               - **Formül**: 
               - **Hata**: O(h).
               - **Avantaj**: İleri fark gibi basit.
               - **Dezavantaj**: Hassasiyet sınırlı.

            3. **Merkezi Fark**:
               - **Formül**: 
               - **Hata**: O(h²), yani hata h² ile orantılı olarak daha hızlı azalır.
               - **Avantaj**: Daha yüksek hassasiyet.
               - **Dezavantaj**: Daha fazla hesaplama gerektirir.
            """)
            st.latex(r"f'(x) \approx \frac{f(x+h) - f(x)}{h} \quad (\text{İleri Fark})")
            st.latex(r"f'(x) \approx \frac{f(x) - f(x-h)}{h} \quad (\text{Geri Fark})")
            st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \quad (\text{Merkezi Fark})")
            st.markdown("""
            ### Hata Analizi
            Hata, adım boyu \( h \)'ye bağlıdır. Merkezi fark, \( O(h^2) \) hata oranıyla daha doğrudur. Ancak, \( h \) çok küçük seçilirse, yuvarlama hataları artabilir.
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fonksiyon ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Fonksiyon**: Matematiksel bir fonksiyon girin (örneğin, `sin(x)`, `x**2`). Desteklenen işlemler: `sin`, `cos`, `exp`, `sqrt`, `log`, `tan`.
            - **Aralık**: Türevlerin hesaplanacağı x aralığını belirleyin.
            - **Nokta Sayısı**: Grafik için kaç nokta kullanılacağını seçin (daha fazla nokta, daha pürüzsüz grafik).
            - **Adım Boyu (h)**: Türev hesaplamalarında kullanılacak küçük artış (örneğin, 0.01). Küçük \( h \), daha doğru sonuçlar verir, ancak çok küçükse yuvarlama hataları artabilir.
            """)

            func_str = st.text_input("Fonksiyon f(x):", "sin(x)", key="deriv_func_str")
            example_funcs = ["sin(x)", "x**2", "exp(x)", "x**3 - 2*x"]
            selected_example = st.selectbox("Örnek Fonksiyon:", [""] + example_funcs, key="deriv_example_select")
            if selected_example:
                func_str = selected_example

            f, expr = self.parse_function_string(func_str)
            if expr:
                st.latex(f"f(x) = {sp.latex(expr)}")
                try:
                    df_expr = sp.diff(expr, sp.symbols('x'))
                    st.latex(f"f'(x) = {sp.latex(df_expr)}")
                except Exception:
                    st.warning("Analitik türev hesaplanamadı.")
            else:
                st.error("Geçerli bir fonksiyon girin (örneğin, 'sin(x)' veya 'x**2').")
                return

            x_start = st.number_input("Başlangıç x:", value=0.0, key="deriv_x_start")
            x_end = st.number_input("Bitiş x:", value=2*np.pi, key="deriv_x_end")
            num_points = st.number_input("Nokta Sayısı:", value=100, min_value=10, step=10, key="deriv_num_points")
            h = st.number_input("Adım Boyu (h):", value=0.01, format="%.3f", key="deriv_h")

        if f is None:
            return

        try:
            x_vals = np.linspace(x_start, x_end, num_points)
            y_vals = f(x_vals)
        except Exception as e:
            st.error(f"Fonksiyon değerlendirme hatası: {e}")
            return

        # Türev hesaplamaları
        forward_deriv = np.array([self.forward_diff(f, x, h) for x in x_vals])
        backward_deriv = np.array([self.backward_diff(f, x, h) for x in x_vals])
        central_deriv = np.array([self.central_diff(f, x, h) for x in x_vals])

        # Analitik türev
        try:
            df_expr = sp.diff(expr, sp.symbols('x'))
            df = sp.lambdify(sp.symbols('x'), df_expr, modules=['numpy'])
            analytical_deriv = df(x_vals)
        except Exception:
            analytical_deriv = None

        with col2:
            st.subheader("Sonuçlar ve Grafik")
            st.markdown("""
            **Grafik Yorumlama**:
            - **İleri/Geri/Merkezi Fark**: Sayısal türevler, fonksiyonun eğimini yaklaşık olarak gösterir.
            - **Analitik Türev**: Varsa, fonksiyonun gerçek türevi ile karşılaştırma sağlar.
            - Grafikte, yöntemlerin ne kadar yakın olduğunu gözlemleyin. Merkezi fark genellikle analitik türeve daha yakındır.
            """)

            try:
                plots = [forward_deriv, backward_deriv, central_deriv]
                labels = ["İleri Fark", "Geri Fark", "Merkezi Fark"]
                if analytical_deriv is not None:
                    plots.append(analytical_deriv)
                    labels.append("Analitik Türev")
                if len(plots) == 0 or len(x_vals) != len(plots[0]):
                    st.error("Grafik çizimi için veri uyumsuz: x ve y verilerinin boyutları eşleşmiyor.")
                else:
                    self.plot_graph(x_vals, plots, xlabel="x", ylabel="f'(x)", title="Sayısal Türevler", labels=labels, figsize=(6, 4))
            except Exception as e:
                st.error(f"Grafik çizme hatası: {e}")

            # Hata Analizi
            if analytical_deriv is not None:
                st.subheader("Hata Analizi")
                errors = {
                    "İleri Fark": np.abs(forward_deriv - analytical_deriv),
                    "Geri Fark": np.abs(backward_deriv - analytical_deriv),
                    "Merkezi Fark": np.abs(central_deriv - analytical_deriv)
                }
                error_data = [{"Yöntem": method, "Ortalama Hata": np.mean(error), "Maksimum Hata": np.max(error)} for method, error in errors.items()]
                st.dataframe(error_data)
                st.markdown("""
                **Hata Yorumlama**:
                - **Ortalama Hata**: Yöntemin genel doğruluğunu gösterir.
                - **Maksimum Hata**: En kötü durumdaki hatayı belirtir.
                - Merkezi farkın hataları genellikle daha küçüktür, çünkü \( O(h^2) \) hata oranına sahiptir.
                """)
                self.plot_graph(x_vals, [errors["İleri Fark"], errors["Geri Fark"], errors["Merkezi Fark"]],
                               xlabel="x", ylabel="Hata", title="Türev Hataları",
                               labels=["İleri Fark", "Geri Fark", "Merkezi Fark"], figsize=(6, 4))

            # Animasyon
            if st.button("Merkezi Fark Animasyonunu Göster", key="central_diff_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, merkezi fark yönteminin nasıl çalıştığını gösterir. Her adımda:
                - Fonksiyonun bir noktasında teğet çizgisi çizilir.
                - Teğet çizgisinin eğimi, merkezi fark türevi olarak hesaplanır.
                - Nokta, x ekseni boyunca hareket ederken eğim değişimini gözlemleyin.
                """)
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    x_plot = np.linspace(x_start, x_end, 200)
                    y_plot = f(x_plot)
                    line, = ax.plot(x_plot, y_plot, label="f(x)")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.set_title("Merkezi Fark Türev Animasyonu")
                    ax.grid(True)
                    ax.legend()

                    def animate(frame):
                        x_i = x_vals[frame % len(x_vals)]
                        slope = self.central_diff(f, x_i, h)
                        tangent = f(x_i) + slope * (x_plot - x_i)
                        ax.clear()
                        ax.plot(x_plot, y_plot, label="f(x)")
                        ax.plot(x_plot, tangent, 'r--', label=f"Türev: {slope:.4f}")
                        ax.scatter([x_i], [f(x_i)], color='red')
                        ax.set_xlabel("x")
                        ax.set_ylabel("f(x)")
                        ax.set_title("Merkezi Fark Türev Animasyonu")
                        ax.grid(True)
                        ax.legend()
                        return line,

                    def init():
                        ax.plot(x_plot, y_plot, label="f(x)")
                        ax.legend()
                        return line,

                    self.save_animation(fig, animate, frames=len(x_vals), init_func=init, interval=200)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Animasyon hatası: {e}")

        self.velocity_acceleration_analysis(f)

    def velocity_acceleration_analysis(self, f):
        st.subheader("Uygulama: Hız ve İvme Analizi")
        st.markdown("""
        Bu bölüm, bir konum fonksiyonu \( s(t) \) verildiğinde, hız (\( v(t) = s'(t) \)) ve ivme (\( a(t) = s''(t) \)) hesaplar.
        **Gerçek Dünya Uygulaması**:
        - **Fizik**: Bir cismin hareketini analiz etmek (örneğin, araba, roket).
        - **Mühendislik**: Titreşim analizi, kontrol sistemleri.
        - **Örnek**: \( s(t) = t^2 \) için, \( v(t) = 2t \), \( a(t) = 2 \).
        """)

        t_start = st.number_input("Başlangıç Zamanı (t0):", value=0.0, key="vel_t_start")
        t_end = st.number_input("Bitiş Zamanı (t_end):", value=10.0, key="vel_t_end")
        num_points = st.number_input("Nokta Sayısı:", value=100, min_value=10, key="vel_num_points")
        h = st.number_input("Adım Boyu (h):", value=0.01, format="%.3f", key="vel_h")

        try:
            t_vals = np.linspace(t_start, t_end, num_points)
            s_vals = f(t_vals)
        except Exception as e:
            st.error(f"Konum fonksiyonu değerlendirme hatası: {e}")
            return

        velocity = np.array([self.central_diff(f, t, h) for t in t_vals])
        def first_deriv(t):
            return self.central_diff(f, t, h)
        acceleration = np.array([self.central_diff(first_deriv, t, h) for t in t_vals])

        try:
            self.plot_graph(t_vals, [s_vals, velocity, acceleration], xlabel="Zaman (t)", ylabel="Değer",
                           title="Konum, Hız ve İvme",
                           labels=["Konum s(t)", "Hız v(t)", "İvme a(t)"], figsize=(6, 4))
            st.markdown("""
            **Grafik Yorumlama**:
            - **Konum s(t)**: Cismin zamanla konumu.
            - **Hız v(t)**: Konumun türevi, hareket hızını gösterir.
            - **İvme a(t)**: Hızın türevi, hızdaki değişimi gösterir.
            Örnek: \( s(t) = \sin(t) \) için, hız \( v(t) = \cos(t) \), ivme \( a(t) = -\sin(t) \).
            """)
        except Exception as e:
            st.error(f"Grafik çizme hatası: {e}")

        # Quiz
        with st.expander("Quiz: Sayısal Türev Bilginizi Test Edin"):
            st.markdown("Aşağıdaki soruyu yanıtlayın:")
            question = st.radio(
                "Hangi yöntem daha hassastır?",
                ["İleri Fark", "Geri Fark", "Merkezi Fark"],
                key="quiz_deriv"
            )
            if st.button("Cevabı Kontrol Et", key="quiz_check"):
                if question == "Merkezi Fark":
                    st.success("Doğru! Merkezi fark, \( O(h^2) \) hata oranıyla daha hassastır.")
                else:
                    st.error("Yanlış. Merkezi fark, \( O(h^2) \) hata oranıyla en hassas yöntemdir.")