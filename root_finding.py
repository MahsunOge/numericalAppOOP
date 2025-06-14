# root_finding.py
import sympy as sp
from base_module import BaseModule
import streamlit as st
import numpy as np
from scipy.optimize import bisect, newton
import time
import matplotlib.pyplot as plt

class RootFindingModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'root_finding_results' not in self.session_state:
            self.session_state.root_finding_results = []

    def run(self):
        st.header("Kök Bulma Yöntemleri")
        st.markdown("""
        Bu modül, \( f(x) = 0 \) denkleminin köklerini bulur:
        - **Bisection (İkiye Bölme)**: Aralığı yarıya bölerek kökü bulur. Sağlam, ancak yavaş (\( O(\log n) \)).
        - **Newton-Raphson**: Türev kullanarak köke hızlı yakınsar (\( O(n^2) \)), ancak iyi bir başlangıç noktası gerekir.
        - **Secant**: Türev yerine iki noktadan geçen sekant çizgisini kullanır, iki başlangıç noktası gerektirir.
        **Uygulama**: Kar-zarar analizi için başabaş noktası hesaplama.
        **Giriş Formatı**: 'x**3 - 2*x - 5' gibi matematiksel ifadeler.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fonksiyon ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Fonksiyon**: Matematiksel bir ifade girin (örn: 'x**3 - 2*x - 5').
            - **Aralık (Bisection)**: \( [a, b] \) aralığında \( f(a) \cdot f(b) < 0 \) olmalı.
            - **Başlangıç Noktaları**: Newton için \( x_0 \), Secant için \( x_0 \) ve \( x_1 \).
            - **Tolerans**: Kökün hassasiyeti (örn: \( 10^{-6} \)).
            """)
            func_str = st.text_input("Fonksiyon (f(x)):", "x**3 - 2*x - 5", key="rf_func_str")
            example_funcs = ["x**3 - 2*x - 5", "x - exp(-x)", "cos(x) - x", "x**2 - 4*sin(x)"]
            selected_example = st.selectbox("Örnek Fonksiyon:", [""] + example_funcs, key="rf_example_select")
            if selected_example:
                func_str = selected_example

            f, expr = self.parse_function_string(func_str)
            if expr:
                st.latex(f"f(x) = {sp.latex(expr)}")
                try:
                    df_expr = sp.diff(expr, sp.symbols('x'))
                    df, _ = self.parse_function_string(str(df_expr))
                    st.latex(f"f'(x) = {sp.latex(df_expr)}")
                except Exception:
                    df = None
                    st.warning("Türev alınamadı. Newton-Raphson için manuel türev gerekebilir.")
            else:
                st.warning("Geçerli bir fonksiyon girin.")
                return

            a_bisection = st.number_input("Bisection Aralığı a:", value=1.0, key="rf_a")
            b_bisection = st.number_input("Bisection Aralığı b:", value=3.0, key="rf_b")
            x0_newton = st.number_input("Newton Başlangıç x0:", value=2.5, key="rf_x0_newton")
            x0_secant = st.number_input("Secant Başlangıç x0:", value=2.0, key="rf_x0_secant")
            x1_secant = st.number_input("Secant Başlangıç x1:", value=3.0, key="rf_x1_secant")
            tol = st.number_input("Tolerans:", value=1e-6, format="%.1e", key="rf_tol")
            max_iter = st.number_input("Maksimum İterasyon:", value=100, step=10, key="rf_max_iter")

        if f is None:
            return

        results_data = []
        iter_steps = {"Bisection": [], "Newton-Raphson": [], "Secant": []}  # Animasyon için adımları kaydet
        plot_data = {'x': np.linspace(min(a_bisection, x0_newton, x0_secant) - 1, max(b_bisection, x0_newton, x1_secant) + 1, 400)}
        plot_data['y'] = f(plot_data['x'])

        # Bisection
        try:
            if f(a_bisection) * f(b_bisection) >= 0:
                st.warning("Bisection için f(a) ve f(b) zıt işaretli olmalı.")
            else:
                a, b = a_bisection, b_bisection
                start_time = time.perf_counter()
                for _ in range(max_iter):
                    c = (a + b) / 2
                    iter_steps["Bisection"].append((a, b, c))
                    if f(c) == 0 or (b - a) / 2 < tol:
                        break
                    if f(c) * f(a) < 0:
                        b = c
                    else:
                        a = c
                root_bisection = c
                time_taken = (time.perf_counter() - start_time) * 1000
                results_data.append({
                    "Yöntem": "Bisection", "Kök": root_bisection,
                    "İterasyon": len(iter_steps["Bisection"]), "f(Kök)": f(root_bisection), "Süre (ms)": time_taken
                })
        except Exception as e:
            results_data.append({"Yöntem": "Bisection", "Hata": str(e)})

        # Newton-Raphson
        if df:
            try:
                x = x0_newton
                start_time = time.perf_counter()
                for i in range(max_iter):
                    fx = f(x)
                    dfx = df(x)
                    iter_steps["Newton-Raphson"].append((x, fx, dfx))
                    if abs(fx) < tol:
                        break
                    if abs(dfx) < 1e-15:
                        raise ValueError("Türev sıfıra çok yakın.")
                    x = x - fx / dfx
                root_newton = x
                time_taken = (time.perf_counter() - start_time) * 1000
                results_data.append({
                    "Yöntem": "Newton-Raphson", "Kök": root_newton,
                    "İterasyon": len(iter_steps["Newton-Raphson"]), "f(Kök)": f(root_newton), "Süre (ms)": time_taken
                })
            except Exception as e:
                results_data.append({"Yöntem": "Newton-Raphson", "Hata": str(e)})
        else:
            st.warning("Newton-Raphson için türev bulunamadı.")

        # Secant
        try:
            x_n_minus_1, x_n = x0_secant, x1_secant
            start_time = time.perf_counter()
            for i in range(max_iter):
                f_xn = f(x_n)
                f_xn_minus_1 = f(x_n_minus_1)
                iter_steps["Secant"].append((x_n_minus_1, x_n, f_xn, f_xn_minus_1))
                if abs(f_xn) < tol:
                    break
                if abs(f_xn - f_xn_minus_1) < 1e-15:
                    st.warning("Secant: f(xn) ve f(xn-1) çok yakın.")
                    break
                x_n_plus_1 = x_n - f_xn * (x_n - x_n_minus_1) / (f_xn - f_xn_minus_1)
                x_n_minus_1, x_n = x_n, x_n_plus_1
            root_secant = x_n
            time_taken = (time.perf_counter() - start_time) * 1000
            results_data.append({
                "Yöntem": "Secant", "Kök": root_secant,
                "İterasyon": len(iter_steps["Secant"]), "f(Kök)": f(root_secant), "Süre (ms)": time_taken
            })
        except Exception as e:
            results_data.append({"Yöntem": "Secant", "Hata": str(e)})

        with col2:
            st.subheader("Sonuçlar")
            st.markdown("""
            **Sonuç Yorumlama**:
            - **Kök**: \( f(x) = 0 \) sağlayan \( x \) değeri.
            - **İterasyon**: Yakınsama için gereken adım sayısı.
            - **f(Kök)**: Kökteki fonksiyon değeri (sıfıra yakın olmalı).
            - **Süre**: Hesaplama süresi (ms).
            """)
            if results_data:
                # Hata sütununu kaldır
                filtered_results = [{k: v for k, v in res.items() if k != "Hata"} for res in results_data]
                st.dataframe(filtered_results)

            st.subheader("Fonksiyon Grafiği")
            points = [(res["Kök"], f(res["Kök"]), f'{res["Yöntem"]} Kökü: {res["Kök"]:.4f}')
                      for res in results_data if "Kök" in res and res["Kök"] is not None]
            self.plot_graph(plot_data['x'], plot_data['y'], xlabel="x", ylabel="f(x)",
                           title="Fonksiyon ve Kökler", labels=["f(x)"], points=points, figsize=(4, 3))

            # Animasyonlar
            if st.button("Kök Bulma Animasyonlarını Göster", key="rf_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyonlar, kök bulma yöntemlerinin adımlarını gösterir:
                - **Bisection**: Aralığı yarıya bölerek köke yaklaşır. Mavi çizgiler aralığı, kırmızı nokta orta noktayı gösterir.
                - **Newton-Raphson**: Teğet çizgisiyle köke yakınsar. Kırmızı nokta mevcut tahmini, mavi çizgi teğeti gösterir.
                - **Secant**: İki noktadan geçen sekant çizgisiyle köke yaklaşır. Kırmızı nokta mevcut tahmini, mavi çizgi sekantı gösterir.
                - Her adımda, iterasyon numarası ve mevcut \( x \) değeri belirtilir.
                """)
                for method in ["Bisection", "Newton-Raphson", "Secant"]:
                    if iter_steps[method]:
                        try:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.plot(plot_data['x'], plot_data['y'], 'b-', label="f(x)")
                            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                            point, = ax.plot([], [], 'ro', label="Mevcut Nokta")
                            line, = ax.plot([], [], 'g-', label="Aralık/Teğet/Sekant")
                            desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                            ax.set_xlabel("x", fontsize=8)
                            ax.set_ylabel("f(x)", fontsize=8)
                            ax.set_title(f"{method} Yöntemi", fontsize=10)
                            ax.legend(fontsize=8)
                            fig.subplots_adjust(bottom=0.25)

                            def animate(i):
                                if i >= len(iter_steps[method]):
                                    return [point, line, desc_text]
                                if method == "Bisection":
                                    a, b, c = iter_steps[method][i]
                                    point.set_data([c], [f(c)])
                                    line.set_data([a, b], [f(a), f(b)])
                                    desc_text.set_text(f"İterasyon {i+1}: x={c:.4f}, Aralık=[{a:.4f}, {b:.4f}]")
                                elif method == "Newton-Raphson":
                                    x, fx, dfx = iter_steps[method][i]
                                    point.set_data([x], [fx])
                                    x_tangent = np.array([x - 0.5, x + 0.5])
                                    y_tangent = fx + dfx * (x_tangent - x)
                                    line.set_data(x_tangent, y_tangent)
                                    desc_text.set_text(f"İterasyon {i+1}: x={x:.4f}")
                                elif method == "Secant":
                                    x_prev, x_curr, f_curr, f_prev = iter_steps[method][i]
                                    point.set_data([x_curr], [f_curr])
                                    line.set_data([x_prev, x_curr], [f_prev, f_curr])
                                    desc_text.set_text(f"İterasyon {i+1}: x={x_curr:.4f}")
                                return [point, line, desc_text]

                            self.save_animation(fig, animate, frames=len(iter_steps[method])+1, interval=500)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"{method} animasyon hatası: {e}")

        self.break_even_analysis(f)

    def break_even_analysis(self, f):
        st.subheader("Uygulama: Kar-Zarar Analizi")
        st.markdown("""
        Bu uygulama, bir işletmenin başabaş noktasını (kar ve zararın eşit olduğu üretim miktarı) hesaplar:
        - **Formül**: Kar = (P - VC) * Q - FC, burada:
          - P: Birim satış fiyatı
          - VC: Birim değişken maliyet
          - FC: Sabit maliyet
          - Q: Üretim miktarı
        - **Amaç**: Kar = 0 denklemini çözerek başabaş noktası \( Q \) bulunur.
        - Sayısal yöntemler (Bisection, Newton-Raphson) kullanılır.
        """)
        fc = st.number_input("Sabit Maliyet (FC):", value=10000.0, step=100.0)
        vc = st.number_input("Birim Başına Değişken Maliyet (VC):", value=5.0, step=0.5)
        p_price = st.number_input("Birim Satış Fiyatı (P):", value=15.0, step=0.5)

        if p_price <= vc:
            st.error("Birim satış fiyatı, birim değişken maliyetten büyük olmalıdır!")
            return

        def profit_func(q_val):
            return (p_price - vc) * q_val - fc

        q_be_analytical = fc / (p_price - vc)
        st.write(f"Analitik Başabaş Noktası: {q_be_analytical:.2f} birim")

        if st.button("Başabaş Noktasını Sayısal Yöntemlerle Bul"):
            q_results = []
            iter_steps_be = {"Bisection": [], "Newton-Raphson": []}  # Animasyon için adımları kaydet
            q_range_max = max(10.0, 3 * q_be_analytical)
            try:
                a, b = 0, q_range_max
                for _ in range(100):
                    c = (a + b) / 2
                    iter_steps_be["Bisection"].append((a, b, c))
                    if profit_func(c) == 0 or (b - a) / 2 < 1e-4:
                        break
                    if profit_func(c) * profit_func(a) < 0:
                        b = c
                    else:
                        a = c
                q_be_bisection = c
                q_results.append({"Yöntem": "Bisection", "Başabaş Miktarı (Q)": q_be_bisection, "İterasyon": len(iter_steps_be["Bisection"])})
            except Exception as e:
                q_results.append({"Yöntem": "Bisection", "Hata": str(e)})

            def profit_func_derivative(q_val):
                return p_price - vc

            try:
                x = q_be_analytical / 2
                for i in range(100):
                    fx = profit_func(x)
                    dfx = profit_func_derivative(x)
                    iter_steps_be["Newton-Raphson"].append((x, fx, dfx))
                    if abs(fx) < 1e-4:
                        break
                    x = x - fx / dfx
                q_be_newton = x
                q_results.append({"Yöntem": "Newton-Raphson", "Başabaş Miktarı (Q)": q_be_newton, "İterasyon": len(iter_steps_be["Newton-Raphson"])})
            except Exception as e:
                q_results.append({"Yöntem": "Newton-Raphson", "Hata": str(e)})

            st.subheader("Başabaş Analizi Sonuçları")
            # Hata sütununu kaldır
            filtered_q_results = [{k: v for k, v in res.items() if k != "Hata"} for res in q_results]
            st.dataframe(filtered_q_results)

            q_plot_vals = np.linspace(0, q_range_max, 200)
            profit_plot_vals = profit_func(q_plot_vals)
            tr_vals = p_price * q_plot_vals
            tc_vals = fc + vc * q_plot_vals
            self.plot_graph(q_plot_vals, [profit_plot_vals, tr_vals, tc_vals], xlabel="Üretim Miktarı (Q)",
                           ylabel="Değer ($)", title="Kar-Zarar Analizi",
                           labels=["Kar", "Toplam Gelir", "Toplam Maliyet"],
                           points=[(q_be_analytical, 0, f'Başabaş Q = {q_be_analytical:.2f}')], figsize=(4, 3))

            # Başabaş animasyonları
            if st.button("Başabaş Analizi Animasyonlarını Göster", key="be_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyonlar, başabaş noktası bulma sürecini gösterir:
                - **Bisection**: Kar fonksiyonu için aralığı yarıya bölerek başabaş noktasına yaklaşır.
                - **Newton-Raphson**: Teğet çizgisiyle başabaş noktasına yakınsar.
                - Kırmızı nokta mevcut tahmini, mavi çizgi aralığı veya teğeti gösterir.
                - Her adımda, iterasyon numarası ve mevcut \( Q \) değeri belirtilir.
                """)
                for method in ["Bisection", "Newton-Raphson"]:
                    if iter_steps_be[method]:
                        try:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.plot(q_plot_vals, profit_plot_vals, 'b-', label="Kar")
                            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                            point, = ax.plot([], [], 'ro', label="Mevcut Nokta")
                            line, = ax.plot([], [], 'g-', label="Aralık/Teğet")
                            desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                            ax.set_xlabel("Q", fontsize=8)
                            ax.set_ylabel("Kar ($)", fontsize=8)
                            ax.set_title(f"{method} Başabaş", fontsize=10)
                            ax.legend(fontsize=8)
                            fig.subplots_adjust(bottom=0.25)

                            def animate(i):
                                if i >= len(iter_steps_be[method]):
                                    return [point, line, desc_text]
                                if method == "Bisection":
                                    a, b, c = iter_steps_be[method][i]
                                    point.set_data([c], [profit_func(c)])
                                    line.set_data([a, b], [profit_func(a), profit_func(b)])
                                    desc_text.set_text(f"İterasyon {i+1}: Q={c:.2f}, Aralık=[{a:.2f}, {b:.2f}]")
                                elif method == "Newton-Raphson":
                                    x, fx, dfx = iter_steps_be[method][i]
                                    point.set_data([x], [fx])
                                    x_tangent = np.array([x - q_range_max/10, x + q_range_max/10])
                                    y_tangent = fx + dfx * (x_tangent - x)
                                    line.set_data(x_tangent, y_tangent)
                                    desc_text.set_text(f"İterasyon {i+1}: Q={x:.2f}")
                                return [point, line, desc_text]

                            self.save_animation(fig, animate, frames=len(iter_steps_be[method])+1, interval=500)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"{method} animasyon hatası: {e}")

    def parse_function_string(self, func_str):
        """Fonksiyon stringini SymPy ifadesine ve çalıştırılabilir fonksiyona çevirir."""
        try:
            x = sp.symbols('x')
            safe_dict = {
                'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp, 'log': sp.log,
                'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E
            }
            expr = sp.sympify(func_str, locals=safe_dict)
            f = sp.lambdify(x, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt}])
            return f, expr
        except Exception:
            return None, None