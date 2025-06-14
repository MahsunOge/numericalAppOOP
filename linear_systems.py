# linear_systems.py
import streamlit as st
import numpy as np
from base_module import BaseModule
import matplotlib.pyplot as plt
import sympy as sp

class LinearSystemsModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'linear_system_results' not in self.session_state:
            self.session_state.linear_system_results = []

    def run(self):
        st.header("Doğrusal Denklem Sistemleri")
        st.markdown("""
        Bu modül, \( Ax = b \) formundaki doğrusal denklem sistemlerini çözer. Dört yöntem sunulmaktadır:
        - **NumPy Solve**: Doğrudan çözüm, hızlı ve kararlı (\( LU \) ayrışımı kullanır).
        - **Gauss Eleme**: Pivotlama ile sağlam, doğrudan yöntem.
        - **Jacobi**: İteratif, diagonal baskın matrisler için uygun.
        - **Gauss-Seidel**: İteratif, daha hızlı yakınsama sağlar.
        Uygulamalar: Elektrik devreleri, yapısal analiz, optimizasyon.
        """)

        # Teori Bölümü
        with st.expander("Teori: Doğrusal Sistem Çözüm Yöntemleri"):
            st.markdown("""
            Doğrusal denklem sistemleri, \( Ax = b \) formundadır:
            - \( A \): \( n \times n \) katsayı matrisi.
            - \( x \): \( n \)-boyutlu bilinmeyen vektör.
            - \( b \): \( n \)-boyutlu sağ taraf vektörü.

            ### Yöntemler
            1. **NumPy Solve**:
               - **Fikir**: \( LU \) ayrışımı ile \( Ax = b \) denklemini çözer.
               - **Hata**: Yuvarlama hataları dışında hassas.
               - **Avantaj**: Hızlı, genel amaçlı.
               - **Dezavantaj**: Matris yapısına özel optimizasyon yok.

            2. **Gauss Eleme**:
               - **Fikir**: Matrisi üst üçgen forma getirir, ardından geri yerine geçirme yapar.
               - **Formül**:
                 - Satır işlemi: \( \text{Satır}_j \gets \text{Satır}_j - \frac{a_{ji}}{a_{ii}} \cdot \text{Satır}_i \)
                 - Geri yerine geçirme: \( x_i = \frac{b_i - \sum_{j=i+1}^n a_{ij} x_j}{a_{ii}} \)
               - **Hata**: Pivotlama ile yuvarlama hataları azalır.
               - **Avantaj**: Sağlam, anlaşılır.
               - **Dezavantaj**: Büyük sistemlerde yavaş.

            3. **Jacobi**:
               - **Fikir**: Her bilinmeyeni iteratif olarak günceller.
               - **Formül**:
                 \[
                 x_i^{(k+1)} = \frac{b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}}{a_{ii}}
                 \]
               - **Hata**: Diagonal baskın matrislerde yakınsar.
               - **Avantaj**: Basit, paralel hesaplama için uygun.
               - **Dezavantaj**: Yavaş yakınsama.

            4. **Gauss-Seidel**:
               - **Fikir**: Güncellenen değerleri hemen kullanır.
               - **Formül**:
                 \[
                 x_i^{(k+1)} = \frac{b_i - \sum_{j<i} a_{ij} x_j^{(k+1)} - \sum_{j>i} a_{ij} x_j^{(k)}}{a_{ii}}
                 \]
               - **Hata**: Jacobi’den daha hızlı yakınsar.
               - **Avantaj**: Daha hızlı, az bellek.
               - **Dezavantaj**: Yakınsama garantisi sınırlı.

            ### Hata Analizi
            - **Kalıntı Normu**: \( \|Ax - b\|_2 \). Düşük norm, çözümü daha doğru gösterir.
            - **Tekillik**: \( \det(A) \approx 0 \) ise sistem çözümsüz veya kararsızdır.
            """)
            st.latex(r"Ax = b")
            st.latex(r"x_i^{(k+1)} = \frac{b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}}{a_{ii}} \quad (\text{Jacobi})")
            st.latex(r"x_i^{(k+1)} = \frac{b_i - \sum_{j<i} a_{ij} x_j^{(k+1)} - \sum_{j>i} a_{ij} x_j^{(k)}}{a_{ii}} \quad (\text{Gauss-Seidel})")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matris ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Matris Boyutu (n)**: \( A \) matrisinin boyutunu seçin (\( n \times n \)).
            - **Katsayı Matrisi A**: Her satırı virgülle ayırarak girin (örneğin, `1, 0, 0`).
            - **Sağ Taraf Vektörü b**: \( n \) elemanlı, virgülle ayrılmış (örneğin, `1, 2, 3`).
            - **Başlangıç Tahmini x0**: Jacobi ve Gauss-Seidel için, \( n \) elemanlı (örneğin, `0, 0, 0`).
            - **Tolerans**: İteratif yöntemlerin durma kriteri (örneğin, \( 10^{-6} \)).
            - **Maksimum İterasyon**: İteratif yöntemler için sınır.
            """)

            n = st.number_input("Matris Boyutu (n x n):", min_value=2, max_value=10, value=3, step=1, key="lin_n")
            
            # Katsayı matrisi A girişi
            st.write("Katsayı Matrisi A (her satırı virgülle ayırarak girin):")
            A_input = []
            for i in range(n):
                default_row = ",".join(["1" if i == j else "0" for j in range(n)])
                row = st.text_input(f"Satır {i+1}:", value=default_row, key=f"lin_A_{i}")
                A_input.append(row)
            
            # Sağ taraf vektörü b girişi
            b_input = st.text_input("Sağ Taraf Vektörü b (virgülle ayrılmış):", value=",".join([str(i+1) for i in range(n)]), key="lin_b")
            
            # Parametreler
            x0_input = st.text_input("Başlangıç Tahmini x0 (Jacobi/Gauss-Seidel, virgülle ayrılmış):", value=",".join(["0"] * n), key="lin_x0")
            tol = st.number_input("Tolerans (iteratif yöntemler):", value=1e-6, format="%.1e", key="lin_tol")
            max_iter = st.number_input("Maksimum İterasyon (iteratif yöntemler):", value=100, step=10, key="lin_max_iter")

            # Matris ve vektörleri ayrıştır
            try:
                A = np.array([[float(x) for x in row.split(',')] for row in A_input])
                b = np.array([float(x.strip()) for x in b_input.split(',')])
                x0 = np.array([float(x.strip()) for x in x0_input.split(',')])
                if A.shape != (n, n) or b.shape != (n,) or x0.shape != (n,):
                    st.error("Matris A (n x n), vektör b ve x0 (n) boyutları uyuşmalı!")
                    return
            except ValueError:
                st.error("Geçerli sayısal değerler girin (örn: 1, 0, 0).")
                return

            # Tekillik kontrolü
            try:
                if np.abs(np.linalg.det(A)) < 1e-10:
                    st.error("Matris A tekil veya neredeyse tekil! Çözüm mümkün olmayabilir.")
                    return
            except np.linalg.LinAlgError:
                st.error("Matris A tekil! Çözüm mümkün değil.")
                return

        # Çözüm yöntemleri
        results = []

        # NumPy Solve
        try:
            x_np = np.linalg.solve(A, b)
            residual_np = np.linalg.norm(A @ x_np - b)
            results.append({"Yöntem": "NumPy Solve", "Çözüm": x_np.tolist(), "Kalıntı": residual_np})
        except np.linalg.LinAlgError as e:
            results.append({"Yöntem": "NumPy Solve", "Hata": str(e)})

        # Gauss Eleme (pivotlama ile)
        def gauss_elimination(A, b):
            A = A.copy().astype(float)
            b = b.copy().astype(float)
            n = len(b)
            steps = []  # Animasyon için adımları kaydet
            for i in range(n):
                # Pivot seçimi
                max_row = i + np.argmax(np.abs(A[i:, i]))
                if A[max_row, i] == 0:
                    raise ValueError("Matris tekil veya pivot sıfır!")
                if max_row != i:
                    A[[i, max_row]], b[[i, max_row]] = A[[max_row, i]], b[[max_row, i]]
                    steps.append((A.copy(), b.copy(), f"Pivot: Satır {i+1} ↔ Satır {max_row+1}"))
                for j in range(i + 1, n):
                    factor = A[j, i] / A[i, i]
                    A[j, i:] -= factor * A[i, i:]
                    b[j] -= factor * b[i]
                    steps.append((A.copy(), b.copy(), f"Elim: Satır {j+1} -= {factor:.2f} × Satır {i+1}"))
            # Geri yerine geçirme
            x = np.zeros(n)
            for i in range(n-1, -1, -1):
                x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
                steps.append((A.copy(), b.copy(), f"Geri: x_{i+1} = {x[i]:.4f}"))
            return x, steps

        try:
            x_gauss, gauss_steps = gauss_elimination(A, b)
            residual_gauss = np.linalg.norm(A @ x_gauss - b)
            results.append({"Yöntem": "Gauss Eleme", "Çözüm": x_gauss.tolist(), "Kalıntı": residual_gauss})
        except Exception as e:
            results.append({"Yöntem": "Gauss Eleme", "Hata": str(e)})
            gauss_steps = []

        # Jacobi Yöntemi
        def jacobi(A, b, x0, tol, max_iter):
            x = x0.copy().astype(float)
            n = len(b)
            iter_count = 0
            for _ in range(max_iter):
                x_new = np.zeros(n)
                for i in range(n):
                    x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
                if np.linalg.norm(x_new - x) < tol:
                    return x_new, iter_count + 1
                x = x_new
                iter_count += 1
            return x, iter_count

        try:
            x_jacobi, iter_jacobi = jacobi(A, b, x0, tol, max_iter)
            residual_jacobi = np.linalg.norm(A @ x_jacobi - b)
            results.append({"Yöntem": "Jacobi", "Çözüm": x_jacobi.tolist(), "Kalıntı": residual_jacobi, "İterasyon": iter_jacobi})
        except Exception as e:
            results.append({"Yöntem": "Jacobi", "Hata": str(e)})

        # Gauss-Seidel Yöntemi
        def gauss_seidel(A, b, x0, tol, max_iter):
            x = x0.copy().astype(float)
            n = len(b)
            iter_count = 0
            for _ in range(max_iter):
                x_new = x.copy()
                for i in range(n):
                    x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
                if np.linalg.norm(x_new - x) < tol:
                    return x_new, iter_count + 1
                x = x_new
                iter_count += 1
            return x, iter_count

        try:
            x_gs, iter_gs = gauss_seidel(A, b, x0, tol, max_iter)
            residual_gs = np.linalg.norm(A @ x_gs - b)
            results.append({"Yöntem": "Gauss-Seidel", "Çözüm": x_gs.tolist(), "Kalıntı": residual_gs, "İterasyon": iter_gs})
        except Exception as e:
            results.append({"Yöntem": "Gauss-Seidel", "Hata": str(e)})

        with col2:
            st.subheader("Sonuçlar ve Görselleştirme")
            st.markdown("""
            **Sonuç Yorumlama**:
            - **Çözüm**: \( x \) vektörü, sistemin bilinmeyenlerini temsil eder.
            - **Kalıntı**: \( \|Ax - b\|_2 \), çözümün doğruluğunu gösterir (düşük = daha doğru).
            - **İterasyon**: Jacobi ve Gauss-Seidel için yakınsama hızını belirtir.
            - Grafik, her yöntemin çözüm vektörünü karşılaştırır.
            """)
            if results:
                st.dataframe(results)
            
            # Çözüm görselleştirme
            if any("Çözüm" in res for res in results):
                x_vals = np.arange(n)
                plots = [res["Çözüm"] for res in results if "Çözüm" in res]
                labels = [res["Yöntem"] for res in results if "Çözüm" in res]
                self.plot_graph(x_vals, plots, xlabel="Bilinmeyen (x_i)", ylabel="Değer", title="Çözüm Vektörleri",
                               labels=labels, figsize=(6, 4))

            # Matris Görselleştirme
            st.subheader("Katsayı Matrisi A")
            try:
                fig, ax = plt.subplots(figsize=(4, 4))
                cax = ax.matshow(A, cmap="viridis")
                for (i, j), val in np.ndenumerate(A):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white")
                fig.colorbar(cax)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Matris görselleştirme hatası: {e}")

            # Karşılaştırma Tablosu
            st.subheader("Yöntem Karşılaştırması")
            comparison = [
                {"Yöntem": "NumPy Solve", "Tür": "Doğrudan", "Hata": "Düşük", "Hız": "Hızlı", "Uygunluk": "Genel"},
                {"Yöntem": "Gauss Eleme", "Tür": "Doğrudan", "Hata": "Düşük", "Hız": "Orta", "Uygunluk": "Küçük sistemler"},
                {"Yöntem": "Jacobi", "Tür": "İteratif", "Hata": "Orta", "Hız": "Yavaş", "Uygunluk": "Diagonal baskın"},
                {"Yöntem": "Gauss-Seidel", "Tür": "İteratif", "Hata": "Orta", "Hız": "Orta", "Uygunluk": "Diagonal baskın"}
            ]
            st.dataframe(comparison)

            # Animasyon
            if st.button("Gauss Eleme Animasyonunu Göster", key="gauss_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, Gauss Eleme yönteminin adımlarını gösterir:
                - **Pivotlama**: En büyük pivotu seçerek sayısal kararlılık sağlanır.
                - **Elimasyon**: Alt satırları sıfırlamak için satır işlemleri yapılır.
                - **Geri Yerine Geçirme**: Üst üçgen matristen \( x \) vektörü hesaplanır.
                Her adımda artırılmış matris \( [A | b] \) gösterilir.
                """)
                if gauss_steps:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        text_artist = ax.text(0.5, 0.5, "", fontsize=10, ha='center', va='center')

                        def animate(i):
                            A_step, b_step, desc = gauss_steps[i]
                            # Matrisi ve vektörü birleştir
                            display_matrix = np.hstack([A_step, b_step.reshape(-1, 1)])
                            # Matris metnini oluştur
                            matrix_text = "\\begin{bmatrix}\n"
                            for row in display_matrix:
                                matrix_text += " & ".join([f"{x:.2f}" for x in row]) + " \\\\\n"
                            matrix_text += "\\end{bmatrix}"
                            # Metni güncelle
                            text_artist.set_text(f"${matrix_text}$\n{desc}")
                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.axis('off')
                            ax.set_title(f"Gauss Eleme: Adım {i+1}")
                            return [text_artist]

                        self.save_animation(fig, animate, frames=len(gauss_steps), interval=1000)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Animasyon hatası: {e}")
                else:
                    st.warning("Gauss Eleme adımları mevcut değil (çözüm başarısız olabilir).")

        # Quiz
        with st.expander("Quiz: Doğrusal Sistemler"):
            st.markdown("Aşağıdaki soruyu yanıtlayın:")
            question = st.radio(
                "Hangi yöntem genellikle daha hızlı yakınsar?",
                ["NumPy Solve", "Gauss Eleme", "Jacobi", "Gauss-Seidel"],
                key="quiz_linear"
            )
            if st.button("Cevabı Kontrol Et", key="quiz_linear_check"):
                if question == "Gauss-Seidel":
                    st.success("Doğru! Gauss-Seidel, güncellenen değerleri hemen kullandığı için genellikle Jacobi’den daha hızlı yakınsar.")
                else:
                    st.error("Yanlış. Gauss-Seidel, güncellenen değerleri hemen kullandığı için genellikle daha hızlı yakınsar.")

        self.circuit_analysis_application()

    def circuit_analysis_application(self):
        st.subheader("Uygulama: Elektrik Devresi Analizi")
        st.markdown("""
        Kirchhoff’un akım yasasına dayalı düğüm analizi ile bir elektrik devresinin düğüm gerilimlerini hesaplar:
        \[
        GV = I
        \]
        - \( G \): İletkenlik matrisi (Siemens cinsinden).
        - \( V \): Düğüm gerilimleri (Volt cinsinden).
        - \( I \): Dış akım kaynakları (Amper cinsinden).
        **Gerçek Dünya Uygulaması**:
        - **Elektronik**: Devre tasarımı, güç dağıtımı.
        - **Mühendislik**: Sensör ağları, kontrol sistemleri.
        Örnek: 3 düğümlü bir devre için \( G \) ve \( I \) girin.
        """)
        st.latex(r"GV = I")

        # Örnek devre: 3 düğüm
        st.write("Örnek devre: 3 düğümlü bir sistem (G matrisi, I vektörü).")
        G_input = [
            st.text_input("İletkenlik Matrisi G Satır 1:", "2, -1, 0", key="circ_G1"),
            st.text_input("İletkenlik Matrisi G Satır 2:", "-1, 3, -1", key="circ_G2"),
            st.text_input("İletkenlik Matrisi G Satır 3:", "0, -1, 2", key="circ_G3")
        ]
        I_input = st.text_input("Akım Vektörü I:", "5, 0, -5", key="circ_I")

        try:
            G = np.array([[float(x) for x in row.split(',')] for row in G_input])
            I = np.array([float(x.strip()) for x in I_input.split(',')])
            if G.shape != (3, 3) or I.shape != (3,):
                st.error("G matrisi 3x3, I vektörü 3 boyutlu olmalı!")
                return
        except ValueError:
            st.error("Geçerli sayısal değerler girin.")
            return

        # Tekillik kontrolü
        try:
            if np.abs(np.linalg.det(G)) < 1e-10:
                st.error("Matris G tekil veya neredeyse tekil! Çözüm mümkün olmayabilir.")
                return
        except np.linalg.LinAlgError:
            st.error("Matris G tekil! Çözüm mümkün değil.")
            return

        # Çözüm
        circuit_results = []
        try:
            V_np = np.linalg.solve(G, I)
            residual_np = np.linalg.norm(G @ V_np - I)
            circuit_results.append({"Yöntem": "NumPy Solve", "Gerilimler (V)": V_np.tolist(), "Kalıntı": residual_np})
        except np.linalg.LinAlgError as e:
            circuit_results.append({"Yöntem": "NumPy Solve", "Hata": str(e)})

        try:
            V_gauss, _ = gauss_elimination(G, I)
            residual_gauss = np.linalg.norm(G @ V_gauss - I)
            circuit_results.append({"Yöntem": "Gauss Eleme", "Gerilimler (V)": V_gauss.tolist(), "Kalıntı": residual_gauss})
        except Exception as e:
            circuit_results.append({"Yöntem": "Gauss Eleme", "Hata": str(e)})

        st.subheader("Devre Analizi Sonuçları")
        st.markdown("""
        **Sonuç Yorumlama**:
        - **Gerilimler (V)**: Her düğümdeki potansiyel farkı gösterir.
        - **Kalıntı**: \( \|GV - I\|_2 \), çözümün doğruluğunu belirtir.
        - Grafik, düğüm gerilimlerini karşılaştırır.
        """)
        if circuit_results:
            st.dataframe(circuit_results)

        # Görselleştirme
        if any("Gerilimler (V)" in res for res in circuit_results):
            x_vals = np.arange(3)
            plots = [res["Gerilimler (V)"] for res in circuit_results if "Gerilimler (V)" in res]
            labels = [res["Yöntem"] for res in circuit_results if "Gerilimler (V)" in res]
            self.plot_graph(x_vals, plots, xlabel="Düğüm", ylabel="Gerilim (V)", title="Düğüm Gerilimleri",
                           labels=labels, figsize=(6, 4))