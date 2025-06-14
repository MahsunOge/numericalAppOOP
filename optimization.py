# optimization.py
import streamlit as st
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from base_module import BaseModule
import matplotlib.pyplot as plt

class OptimizationModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'optimization_results' not in self.session_state:
            self.session_state.optimization_results = []

    def validate_function(self, func_str, opt_type, n_vars=None):
        """Fonksiyon stringini optimizasyon türüne göre doğrula."""
        if opt_type == "Skaler (Tek Değişkenli)":
            if 'x[' in func_str:
                return False, "Skaler fonksiyon sadece 'x' kullanmalı, örneğin: 'x**2' veya 'np.sin(x)'."
            return True, ""
        else:
            for i in range(n_vars):
                if f'x[{i}]' not in func_str and 'np.sum(x**2)' not in func_str:
                    return False, f"Çok değişkenli fonksiyon 'x[0]', 'x[1]', ... kullanmalı, örneğin: 'x[0]**2 + x[1]**2'."
            return True, ""

    def run(self):
        st.header("Optimizasyon Teknikleri")
        st.markdown("""
        Bu modül, amaç fonksiyonlarını minimize veya maksimize eder:  
        - **Skaler Optimizasyon**: Tek değişkenli problemler için (`scipy.optimize.minimize_scalar`).  
        - **Çok Değişkenli Optimizasyon**: Birden fazla değişken için (`scipy.optimize.minimize`).  
        **Yöntemler**: Brent (skaler), Nelder-Mead, BFGS, L-BFGS-B, SLSQP (kısıtlı).  
        **Uygulama**: Maliyet minimizasyonu.  
        **Not**: Maksimizasyon için \(-f(x)\) minimize edilir.  
        **Giriş Formatı**: Skaler için 'x**2', çok değişkenli için 'x[0]**2 + x[1]**2'.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fonksiyon ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Optimizasyon Türü**: Skaler (tek değişken) veya çok değişkenli seçin.
            - **Amaç**: Minimize veya maksimize.
            - **Fonksiyon**: Skaler için 'x**2', çok değişkenli için 'x[0]**2 + x[1]**2' gibi girin.
            - **Sınırlar**: Skaler için alt ve üst sınır, çok değişkenli için [(alt, üst), ...].
            """)
            opt_type = st.radio("Optimizasyon Türü:", ["Skaler (Tek Değişkenli)", "Çok Değişkenli"], key="opt_type")
            minimize_or_maximize = st.radio("Amaç:", ["Minimize", "Maksimize"], key="opt_goal")

            if opt_type == "Skaler (Tek Değişkenli)":
                func_str = st.text_input("Amaç Fonksiyonu f(x):", "x**2 + 2*x + 1", key="opt_func_scalar")
                example_funcs = ["x**2 + 2*x + 1", "np.sin(x)", "np.exp(x) - x", "x**4 - 4*x**2"]
                selected_example = st.selectbox("Örnek Fonksiyon:", [""] + example_funcs, key="opt_example_scalar")
                if selected_example:
                    func_str = selected_example

                a = st.number_input("Alt Sınır (a):", value=-5.0, key="opt_a")
                b = st.number_input("Üst Sınır (b):", value=5.0, key="opt_b")
                method = st.selectbox("Yöntem:", ["brent", "bounded"], key="opt_method_scalar")
            else:
                func_str = st.text_input("Amaç Fonksiyonu f(x0, x1, ...):", "x[0]**2 + x[1]**2", key="opt_func_multi")
                example_funcs = ["x[0]**2 + x[1]**2", "x[0]**2 + 2*x[1]**2 - 2*x[0]*x[1]", "np.sum(x**2)"]
                selected_example = st.selectbox("Örnek Fonksiyon:", [""] + example_funcs, key="opt_example_multi")
                if selected_example:
                    func_str = selected_example

                n_vars = st.number_input("Değişken Sayısı:", min_value=2, max_value=10, value=2, step=1, key="opt_n_vars")
                x0_input = st.text_input("Başlangıç Tahmini x0 (virgülle ayrılmış):", value="0, 0", key="opt_x0")
                bounds_input = st.text_input("Sınırlar [(alt, üst), ...]:", value="(-5, 5), (-5, 5)", key="opt_bounds")
                method = st.selectbox("Yöntem:", ["Nelder-Mead", "BFGS", "L-BFGS-B", "SLSQP"], key="opt_method_multi")

            # Fonksiyon doğrulama
            is_valid, error_msg = self.validate_function(func_str, opt_type, n_vars if opt_type == "Çok Değişkenli" else None)
            if not is_valid:
                st.error(error_msg)
                return

            # Fonksiyonu ayrıştır
            try:
                safe_globals = {"__builtins__": {}, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}
                if opt_type == "Skaler (Tek Değişkenli)":
                    f = lambda x: eval(func_str, safe_globals, {"x": float(x)})
                    if minimize_or_maximize == "Maksimize":
                        f = lambda x: -eval(func_str, safe_globals, {"x": float(x)})
                    f(0)  # Test değerlendirmesi
                else:
                    f = lambda x: eval(func_str, safe_globals, {"x": np.array(x)})
                    if minimize_or_maximize == "Maksimize":
                        f = lambda x: -eval(func_str, safe_globals, {"x": np.array(x)})
                    f(np.zeros(n_vars))  # Test değerlendirmesi
            except NameError as e:
                st.error(f"Fonksiyon formatı hatalı: {e}. {error_msg}")
                return
            except Exception as e:
                st.error(f"Fonksiyon değerlendirme hatası: {e}. Skaler için 'x**2', çok değişkenli için 'x[0]**2 + x[1]**2' kullanın.")
                return

            # Çok değişkenli için ek girişler
            if opt_type == "Çok Değişkenli":
                try:
                    x0 = np.array([float(x.strip()) for x in x0_input.split(',')])
                    if len(x0) != n_vars:
                        st.error(f"Başlangıç tahmini x0, {n_vars} eleman içermeli!")
                        return
                    bounds = eval(f"[{bounds_input}]", {"__builtins__": {}})
                    if len(bounds) != n_vars or not all(len(b) == 2 for b in bounds):
                        st.error(f"Sınırlar {n_vars} adet (alt, üst) çifti içermeli!")
                        return
                except Exception as e:
                    st.error(f"Başlangıç tahmini veya sınırlar hatalı: {e}.")
                    return
            else:
                x0, bounds = None, (a, b)

        # Optimizasyon
        results = []
        iter_points = []  # Animasyon için noktaları kaydet
        if opt_type == "Skaler (Tek Değişkenli)":
            try:
                res = minimize_scalar(f, bounds=bounds, method=method)
                if res.success:
                    results.append({
                        "Yöntem": method,
                        "Optimum x": res.x,
                        "f(x)": -res.fun if minimize_or_maximize == "Maksimize" else res.fun,
                        "Başarı": "Başarılı"
                    })
                else:
                    results.append({"Yöntem": method, "Hata": res.message})
            except Exception as e:
                results.append({"Yöntem": method, "Hata": str(e)})
        else:
            def callback(xk):
                iter_points.append(xk.copy())  # Her iterasyonda noktayı kaydet
            try:
                res = minimize(f, x0, method=method, bounds=bounds, callback=callback)
                if res.success:
                    results.append({
                        "Yöntem": method,
                        "Optimum x": res.x.tolist(),
                        "f(x)": -res.fun if minimize_or_maximize == "Maksimize" else res.fun,
                        "Başarı": "Başarılı",
                        "İterasyon": res.nit if hasattr(res, 'nit') else len(iter_points)
                    })
                else:
                    results.append({"Yöntem": method, "Hata": res.message})
            except Exception as e:
                results.append({"Yöntem": method, "Hata": str(e)})

        with col2:
            st.subheader("Sonuçlar ve Grafik")
            st.markdown("""
            **Sonuç Yorumlama**:
            - **Optimum x**: Fonksiyonun minimum/maksimum olduğu nokta.
            - **f(x)**: Optimum noktadaki fonksiyon değeri.
            - **İterasyon**: Algoritmanın yakınsaması için gereken adım sayısı.
            - Grafik, fonksiyonun şeklini ve optimum noktayı gösterir.
            """)
            if results:
                st.dataframe(results)

            # Görselleştirme
            if opt_type == "Skaler (Tek Değişkenli)":
                x_vals = np.linspace(a, b, 100)
                y_vals = [f(x) if minimize_or_maximize == "Minimize" else -f(x) for x in x_vals]
                points = [(res["Optimum x"], res["f(x)"], f"{res['Yöntem']} Optimum: {res['Optimum x']:.4f}")
                          for res in results if "Optimum x" in res]
                self.plot_graph(x_vals, y_vals, xlabel="x", ylabel="f(x)", title="Amaç Fonksiyonu ve Optimum",
                               labels=["f(x)"], points=points, figsize=(4, 3))
            else:
                if n_vars == 2:  # 2D görselleştirme
                    x1_vals = np.linspace(bounds[0][0], bounds[0][1], 50)
                    x2_vals = np.linspace(bounds[1][0], bounds[1][1], 50)
                    X1, X2 = np.meshgrid(x1_vals, x2_vals)
                    Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    contour = ax.contourf(X1, X2, Z, levels=20, cmap="viridis")
                    fig.colorbar(contour, label="f(x0, x1)", shrink=0.8)
                    for res in results:
                        if "Optimum x" in res:
                            ax.plot(res["Optimum x"][0], res["Optimum x"][1], 'ro', label=f"{res['Yöntem']} Optimum")
                    ax.set_xlabel("x0", fontsize=8)
                    ax.set_ylabel("x1", fontsize=8)
                    ax.set_title("Amaç Fonksiyonu", fontsize=10)
                    ax.legend(fontsize=8)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Görselleştirme sadece 2 değişken için destekleniyor.")

            # Animasyon (sadece çok değişkenli ve 2 değişken için)
            if opt_type == "Çok Değişkenli" and n_vars == 2 and iter_points:
                if st.button("Optimizasyon Animasyonunu Göster", key="opt_anim"):
                    st.markdown("""
                    **Animasyon Açıklaması**:
                    Bu animasyon, çok değişkenli optimizasyon sürecini gösterir:
                    - Amaç fonksiyonu, 2D kontur grafiği olarak görselleştirilir.
                    - Kırmızı nokta, algoritmanın her iterasyonda test ettiği noktayı temsil eder.
                    - Mavi çizgi, algoritmanın izlediği yolu gösterir.
                    - Her adımda, iterasyon numarası ve mevcut \( x \) vektörü belirtilir.
                    """)
                    try:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        x1_vals = np.linspace(bounds[0][0], bounds[0][1], 50)
                        x2_vals = np.linspace(bounds[1][0], bounds[1][1], 50)
                        X1, X2 = np.meshgrid(x1_vals, x2_vals)
                        Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])
                        contour = ax.contourf(X1, X2, Z, levels=20, cmap="viridis")
                        fig.colorbar(contour, label="f(x0, x1)", shrink=0.8)
                        point, = ax.plot([], [], 'ro', label="Mevcut Nokta")
                        path, = ax.plot([], [], 'b-', label="Yol")
                        desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                        ax.set_xlabel("x0", fontsize=8)
                        ax.set_ylabel("x1", fontsize=8)
                        ax.set_title(f"{method} Optimizasyonu", fontsize=10)
                        ax.legend(fontsize=8)
                        fig.subplots_adjust(bottom=0.25)

                        def animate(i):
                            if i < len(iter_points):
                                x = iter_points[i]
                                point.set_data([x[0]], [x[1]])
                                path.set_data([p[0] for p in iter_points[:i+1]], [p[1] for p in iter_points[:i+1]])
                                desc_text.set_text(f"İterasyon {i+1}: x=[{x[0]:.4f}, {x[1]:.4f}]")
                            else:
                                x = res.x
                                point.set_data([x[0]], [x[1]])
                                path.set_data([p[0] for p in iter_points] + [x[0]], [p[1] for p in iter_points] + [x[1]])
                                desc_text.set_text(f"Sonuç: x=[{x[0]:.4f}, {x[1]:.4f}]")
                            return [point, path, desc_text]

                        self.save_animation(fig, animate, frames=len(iter_points)+1, interval=500)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Animasyon hatası: {e}")

        self.cost_minimization_application()

    def cost_minimization_application(self):
        st.subheader("Uygulama: Maliyet Minimizasyonu")
        st.markdown("""
        Bir üretim sürecinde maliyet fonksiyonunu minimize edelim.  
        **Örnek**: \( f(x_0, x_1) = 2x_0^2 + x_1^2 \), burada \( x_0 \): üretim miktarı, \( x_1 \): iş gücü.  
        **Amaç**: Maliyeti en aza indiren \( x_0 \) ve \( x_1 \) değerlerini bulmak.
        """)

        func_str = st.text_input("Maliyet Fonksiyonu (örn: 2*x[0]**2 + x[1]**2):", "2*x[0]**2 + x[1]**2", key="cost_func")
        x0_input = st.text_input("Başlangıç Tahmini (x0, x1):", "1, 1", key="cost_x0")
        bounds_input = st.text_input("Sınırlar [(alt, üst), ...]:", "(0, 10), (0, 10)", key="cost_bounds")
        method = st.selectbox("Yöntem:", ["Nelder-Mead", "BFGS", "L-BFGS-B", "SLSQP"], key="cost_method")

        # Fonksiyon doğrulama
        is_valid, error_msg = self.validate_function(func_str, "Çok Değişkenli", n_vars=2)
        if not is_valid:
            st.error(error_msg)
            return

        try:
            safe_globals = {"__builtins__": {}, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp}
            f = lambda x: eval(func_str, safe_globals, {"x": np.array(x)})
            x0 = np.array([float(x.strip()) for x in x0_input.split(',')])
            bounds = eval(f"[{bounds_input}]", {"__builtins__": {}})
            if len(x0) != 2 or len(bounds) != 2 or not all(len(b) == 2 for b in bounds):
                st.error("2 değişkenli fonksiyon, başlangıç tahmini ve sınırlar gerekli!")
                return
        except Exception as e:
            st.error(f"Giriş hatası: {e}. Örnek: 'x[0]**2 + x[1]**2', x0: '1,1', sınırlar: '(0,10),(0,10)'.")
            return

        cost_results = []
        iter_points_cost = []  # Animasyon için noktaları kaydet
        def callback(xk):
            iter_points_cost.append(xk.copy())
        try:
            res = minimize(f, x0, method=method, bounds=bounds, callback=callback)
            if res.success:
                cost_results.append({
                    "Yöntem": method,
                    "Optimum x": res.x.tolist(),
                    "Maliyet": res.fun,
                    "Başarı": "Başarılı",
                    "İterasyon": res.nit if hasattr(res, 'nit') else len(iter_points_cost)
                })
            else:
                cost_results.append({"Yöntem": method, "Hata": res.message})
        except Exception as e:
            cost_results.append({"Yöntem": method, "Hata": str(e)})

        st.subheader("Maliyet Minimizasyonu Sonuçları")
        st.markdown("""
        **Sonuç Yorumlama**:
        - **Optimum x**: Minimum maliyeti sağlayan üretim miktarı (\( x_0 \)) ve iş gücü (\( x_1 \)).
        - **Maliyet**: Optimum noktadaki maliyet değeri.
        - Grafik, maliyet fonksiyonunu ve optimum noktayı gösterir.
        """)
        if cost_results:
            st.dataframe(cost_results)

        # Görselleştirme (küçültülmüş grafik)
        if cost_results and "Optimum x" in cost_results[0]:
            x1_vals = np.linspace(bounds[0][0], bounds[0][1], 50)
            x2_vals = np.linspace(bounds[1][0], bounds[1][1], 50)
            X1, X2 = np.meshgrid(x1_vals, x2_vals)
            Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])
            fig, ax = plt.subplots(figsize=(4, 3))
            contour = ax.contourf(X1, X2, Z, levels=20, cmap="viridis")
            fig.colorbar(contour, label="Maliyet", shrink=0.8)
            ax.plot(cost_results[0]["Optimum x"][0], cost_results[0]["Optimum x"][1], 'ro', label="Optimum")
            ax.set_xlabel("x0 (Üretim)", fontsize=8)
            ax.set_ylabel("x1 (İş Gücü)", fontsize=8)
            ax.set_title("Maliyet Fonksiyonu", fontsize=10)
            ax.legend(fontsize=8)
            fig.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
            st.pyplot(fig)
            plt.close(fig)

        # Animasyon (maliyet minimizasyonu için)
        if iter_points_cost:
            if st.button("Maliyet Optimizasyonu Animasyonunu Göster", key="cost_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, maliyet fonksiyonunun optimizasyon sürecini gösterir:
                - Maliyet fonksiyonu, 2D kontur grafiği olarak görselleştirilir.
                - Kırmızı nokta, algoritmanın her iterasyonda test ettiği noktayı temsil eder.
                - Mavi çizgi, algoritmanın izlediği yolu gösterir.
                - Her adımda, iterasyon numarası ve mevcut \( x \) vektörü belirtilir.
                """)
                try:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    x1_vals = np.linspace(bounds[0][0], bounds[0][1], 50)
                    x2_vals = np.linspace(bounds[1][0], bounds[1][1], 50)
                    X1, X2 = np.meshgrid(x1_vals, x2_vals)
                    Z = np.array([[f(np.array([x1, x2])) for x1 in x1_vals] for x2 in x2_vals])
                    contour = ax.contourf(X1, X2, Z, levels=20, cmap="viridis")
                    fig.colorbar(contour, label="Maliyet", shrink=0.8)
                    point, = ax.plot([], [], 'ro', label="Mevcut Nokta")
                    path, = ax.plot([], [], 'b-', label="Yol")
                    desc_text = ax.text(0.5, -0.2, "", ha='center', va='top', fontsize=8, transform=ax.transAxes)
                    ax.set_xlabel("x0 (Üretim)", fontsize=8)
                    ax.set_ylabel("x1 (İş Gücü)", fontsize=8)
                    ax.set_title(f"{method} Optimizasyonu", fontsize=10)
                    ax.legend(fontsize=8)
                    fig.subplots_adjust(bottom=0.25)

                    def animate(i):
                        if i < len(iter_points_cost):
                            x = iter_points_cost[i]
                            point.set_data([x[0]], [x[1]])
                            path.set_data([p[0] for p in iter_points_cost[:i+1]], [p[1] for p in iter_points_cost[:i+1]])
                            desc_text.set_text(f"İterasyon {i+1}: x=[{x[0]:.4f}, {x[1]:.4f}]")
                        else:
                            x = res.x
                            point.set_data([x[0]], [x[1]])
                            path.set_data([p[0] for p in iter_points_cost] + [x[0]], [p[1] for p in iter_points_cost] + [x[1]])
                            desc_text.set_text(f"Sonuç: x=[{x[0]:.4f}, {x[1]:.4f}]")
                        return [point, path, desc_text]

                    self.save_animation(fig, animate, frames=len(iter_points_cost)+1, interval=500)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Animasyon hatası: {e}")