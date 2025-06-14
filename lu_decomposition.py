# lu_decomposition.py
import streamlit as st
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from base_module import BaseModule
import matplotlib.pyplot as plt

class LUDecompositionModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'lu_results' not in self.session_state:
            self.session_state.lu_results = []

    def run(self):
        st.header("LU Ayrıştırması ve Çözüm")
        st.markdown("""
        Bu modül, \( Ax = b \) formundaki doğrusal sistemleri LU ayrıştırması ile çözer:  
        - **LU Ayrıştırması**: Matris \( A \)'yı bir kez \( L \) (alt üçgen) ve \( U \) (üst üçgen) matrislerine ayırır (\( O(n^3) \)).  
        - **Çözüm**: \( L \) ve \( U \) kullanılarak her \( b_i \) için hızlı çözüm (\( O(n^2) \)).  
        - **SciPy Kullanımı**: `scipy.linalg.lu_factor` ve `scipy.linalg.lu_solve`.  
        **Uygulama**: Sabit sistemde birden fazla yük senaryosu (örn. devre analizi).  
        **Avantaj**: Tekrarlanan çözümler için verimlilik.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Matris ve Parametreler")
            st.markdown("""
            **Talimatlar**:
            - **Matris Boyutu (n)**: \( A \) matrisinin boyutunu seçin (\( n \times n \)).
            - **Katsayı Matrisi A**: Her satırı virgülle ayırarak girin (örn: `2, -1, 0`).
            - **Sağ Taraf Vektörleri bᵢ**: Her vektörü virgülle, vektörleri noktalı virgülle ayırın (örn: `1,0,0;0,1,0`).
            """)
            n = st.number_input("Matris Boyutu (n x n):", min_value=2, max_value=5, value=3, step=1, key="lu_n")
            
            # Katsayı matrisi A girişi
            st.write("Katsayı Matrisi A (her satırı virgülle ayırarak girin):")
            A_input = []
            for i in range(n):
                row = st.text_input(f"Satır {i+1}:", 
                                   value="2, -1, 0" if i == 0 else "-1, 2, -1" if i == 1 else "0, -1, 2", 
                                   key=f"lu_A_{i}")
                A_input.append(row)
            
            # Birden fazla b vektörü girişi
            st.write("Sağ Taraf Vektörleri bᵢ (her vektörü virgülle, vektörleri noktalı virgülle ayırın):")
            b_input = st.text_area("b Vektörleri (örn: 1,0,0;0,1,0):", value="1,0,0;0,1,0;0,0,1", key="lu_b")
            
            # Matris ve vektörleri ayrıştır
            try:
                A = np.array([[float(x) for x in row.split(',')] for row in A_input])
                b_vectors = [np.array([float(x.strip()) for x in vec.split(',')]) for vec in b_input.split(';')]
                if A.shape != (n, n) or any(b.shape != (n,) for b in b_vectors):
                    st.error("Matris A (n x n) ve her bᵢ vektörü (n) boyutunda olmalı!")
                    return
            except ValueError:
                st.error("Geçerli sayısal değerler girin (örn: 2,-1,0).")
                return

            # Tekillik kontrolü
            try:
                if np.abs(np.linalg.det(A)) < 1e-10:
                    st.error("Matris A tekil veya neredeyse tekil! Çözüm mümkün olmayabilir.")
                    return
            except np.linalg.LinAlgError:
                st.error("Matris A tekil! Çözüm mümkün değil.")
                return

        # LU Ayrıştırması ve Çözüm
        results = []
        lu_steps = []
        try:
            # LU ayrıştırması
            lu, piv = lu_factor(A)  # SciPy’nin LU ayrıştırması
            L = np.tril(lu, k=-1) + np.eye(n)  # L matrisi
            U = np.triu(lu)  # U matrisi
            lu_steps.append((L.copy(), U.copy(), "Başlangıç: L ve U matrisleri oluşturuldu", np.zeros(n), np.zeros(n)))
            
            # Çözüm adımları
            for i, b in enumerate(b_vectors):
                # İleri dönüşüm: Ly = b
                y = np.zeros(n)
                for j in range(n):
                    y[j] = (b[j] - np.dot(L[j, :j], y[:j])) / L[j, j]
                    lu_steps.append((L.copy(), U.copy(), f"İleri Dönüşüm (b_{i+1}): y_{j+1} = {y[j]:.4f}", y.copy(), np.zeros(n)))
                
                # Geri dönüşüm: Ux = y
                x = np.zeros(n)
                for j in range(n-1, -1, -1):
                    x[j] = (y[j] - np.dot(U[j, j+1:], x[j+1:])) / U[j, j]
                    lu_steps.append((L.copy(), U.copy(), f"Geri Dönüşüm (b_{i+1}): x_{j+1} = {x[j]:.4f}", y.copy(), x.copy()))
                
                residual = np.linalg.norm(A @ x - b)
                results.append({"b Vektörü": i+1, "Çözüm (x)": x.tolist(), "Kalıntı": residual})

        except np.linalg.LinAlgError as e:
            results.append({"Yöntem": "LU Ayrıştırması", "Hata": str(e)})

        # NumPy Solve ile karşılaştırma
        try:
            for i, b in enumerate(b_vectors):
                x_np = np.linalg.solve(A, b)
                residual_np = np.linalg.norm(A @ x_np - b)
                results.append({"b Vektörü": i+1, "Yöntem": "NumPy Solve", "Çözüm (x)": x_np.tolist(), "Kalıntı": residual_np})
        except np.linalg.LinAlgError as e:
            results.append({"Yöntem": "NumPy Solve", "Hata": str(e)})

        with col2:
            st.subheader("Sonuçlar ve Görselleştirme")
            st.markdown("""
            **Sonuç Yorumlama**:
            - **Çözüm (x)**: Her \( b_i \) için hesaplanan \( x \) vektörü.
            - **Kalıntı**: \( \|Ax - b\|_2 \), çözümün doğruluğunu gösterir (düşük = daha doğru).
            - Grafik, her \( b_i \) için çözüm vektörlerini karşılaştırır.
            """)
            if results:
                st.dataframe(results)

            # Çözüm görselleştirme
            if any("Çözüm (x)" in res for res in results):
                x_vals = np.arange(n)
                plots = [res["Çözüm (x)"] for res in results if "Çözüm (x)" in res]
                labels = [f"{res['Yöntem'] if 'Yöntem' in res else 'LU'} (b_{res['b Vektörü']})" for res in results if "Çözüm (x)" in res]
                self.plot_graph(x_vals, plots, xlabel="Bilinmeyen (x_i)", ylabel="Değer", title="Çözüm Vektörleri",
                               labels=labels, figsize=(4, 3))

            # Animasyon
            if st.button("LU Ayrıştırması Animasyonunu Göster", key="lu_anim"):
                st.markdown("""
                **Animasyon Açıklaması**:
                Bu animasyon, LU ayrıştırması ve çözüm sürecini gösterir:
                - **Ayrıştırma**: Matris \( A \), \( L \) (alt üçgen) ve \( U \) (üst üçgen) matrislerine ayrılır.
                - **İleri Dönüşüm**: \( Ly = b \) denklemi çözülerek \( y \) vektörü bulunur.
                - **Geri Dönüşüm**: \( Ux = y \) denklemi çözülerek \( x \) vektörü bulunur.
                - Sol panelde \( L \) ve \( U \) matrislerinin oluşumu, sağ panelde \( y \) ve \( x \) vektörlerinin değişimi gösterilir.
                - Her adımda yapılan işlem, sağ panelde açıklanır.
                """)
                if lu_steps:
                    try:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [1.5, 1]})
                        # L ve U heatmap
                        L, U, _, y, x = lu_steps[0]
                        im_L = ax1.imshow(L, cmap='viridis', vmin=np.min([L, U]), vmax=np.max([L, U]))
                        im_U = ax1.imshow(U, cmap='viridis', vmin=np.min([L, U]), vmax=np.max([L, U]), alpha=0)
                        texts_L = []
                        for i in range(n):
                            row = []
                            for j in range(n):
                                text = ax1.text(j, i, f"{L[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
                                row.append(text)
                            texts_L.append(row)
                        texts_U = []
                        for i in range(n):
                            row = []
                            for j in range(n):
                                text = ax1.text(j, i, f"{U[i, j]:.1f}", ha="center", va="center", color="white", alpha=0, fontsize=8)
                                row.append(text)
                            texts_U.append(row)
                        ax1.set_title("L ve U", fontsize=10)
                        ax1.set_xticks([])
                        ax1.set_yticks([])
                        fig.colorbar(im_L, ax=ax1, shrink=0.8, label='Değer')
                        # Çubuk grafiği
                        x_indices = np.arange(n)
                        bars = ax2.bar(x_indices, np.zeros(n), color='skyblue')
                        max_b = np.max(np.abs(b_vectors)) + 1
                        ax2.set_ylim(-max_b, max_b)
                        ax2.set_xticks(x_indices)
                        ax2.set_xticklabels([f"x_{i+1}" for i in range(n)], fontsize=8)
                        ax2.set_ylabel("Değer", fontsize=8)
                        ax2.set_title("Vektör", fontsize=10)
                        # Açıklama metni
                        desc_text = ax2.text(0.5, -0.3, "", ha='center', va='top', fontsize=8, transform=ax2.transAxes)
                        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.9)

                        def animate(i):
                            L, U, desc, y, x = lu_steps[i]
                            # L ve U heatmap
                            im_L.set_array(L)
                            im_U.set_array(U)
                            alpha_L = 1 if "İleri" in desc or "Geri" in desc else 0.5
                            alpha_U = 1 if "Geri" in desc else 0.5
                            im_L.set_alpha(alpha_L)
                            im_U.set_alpha(alpha_U)
                            for r in range(n):
                                for c in range(n):
                                    texts_L[r][c].set_text(f"{L[r, c]:.1f}")
                                    texts_L[r][c].set_alpha(alpha_L)
                                    texts_U[r][c].set_text(f"{U[r, c]:.1f}")
                                    texts_U[r][c].set_alpha(alpha_U)
                            # Çubuk grafiği
                            vec = y if "İleri" in desc else x
                            for bar, val in zip(bars, vec):
                                bar.set_height(val)
                            ax2.set_title("y Vektörü" if "İleri" in desc else "x Vektörü" if "Geri" in desc else "Başlangıç", fontsize=10)
                            # Açıklama
                            desc_text.set_text(desc)
                            return [im_L, im_U, desc_text] + list(bars) + [text for row in texts_L for text in row] + [text for row in texts_U for text in row]

                        self.save_animation(fig, animate, frames=len(lu_steps), interval=1000)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Animasyon hatası: {e}")
                else:
                    st.warning("LU ayrıştırma adımları mevcut değil (çözüm başarısız olabilir).")

        self.load_analysis_application()

    def load_analysis_application(self):
        st.subheader("Uygulama: Çoklu Yük Senaryoları (Yapısal Analiz)")
        st.markdown("""
        Sabit bir sistem (\( K \)) ile birden fazla yük vektörü (\( F_i \)) için çözüm.  
        **Örnek**: Yapısal bir çerçevedeki yer değiştirmeler (\( Ku = F \)).  
        **Amaç**: Farklı yük senaryoları altında düğüm yer değiştirmelerini hesaplamak.
        """)

        # Örnek sistem: 3x3 matris
        st.write("Örnek sistem: 3x3 sertlik matrisi ve yük vektörleri.")
        K_input = [
            st.text_input("Sertlik Matrisi K Satır 1:", "4, -1, 0", key="load_K1"),
            st.text_input("Sertlik Matrisi K Satır 2:", "-1, 4, -1", key="load_K2"),
            st.text_input("Sertlik Matrisi K Satır 3:", "0, -1, 4", key="load_K3")
        ]
        F_input = st.text_area("Yük Vektörleri Fᵢ (örn: 10,0,0;0,10,0):", value="10,0,0;0,10,0;0,0,10", key="load_F")

        try:
            K = np.array([[float(x) for x in row.split(',')] for row in K_input])
            F_vectors = [np.array([float(x.strip()) for x in vec.split(',')]) for vec in F_input.split(';')]
            if K.shape != (3, 3) or any(F.shape != (3,) for F in F_vectors):
                st.error("Matrisi K (3x3) ve her Fᵢ vektörü (3) boyutunda olmalı!")
                return
        except ValueError:
            st.error("Geçerli sayısal değerler girin.")
            return

        # LU ile çözüm
        load_results = []
        try:
            lu, piv = lu_factor(K)
            for i, F in enumerate(F_vectors):
                u = lu_solve((lu, piv), F)
                residual = np.linalg.norm(K @ u - F)
                load_results.append({"Yük Senaryosu": i+1, "Yer Değiştirme (u)": u.tolist(), "Kalıntı": residual})
        except np.linalg.LinAlgError as e:
            load_results.append({"Yöntem": "LU Ayrıştırması", "Hata": str(e)})

        st.subheader("Yük Analizi Sonuçları")
        st.markdown("""
        **Sonuç Yorumlama**:
        - **Yer Değiştirme (u)**: Her yük senaryosu için düğüm yer değiştirmeleri.
        - **Kalıntı**: \( \|Ku - F\|_2 \), çözümün doğruluğunu gösterir.
        - Grafik, farklı yük senaryolarındaki yer değiştirmeleri karşılaştırır.
        """)
        if load_results:
            st.dataframe(load_results)

        # Görselleştirme (küçültülmüş grafik)
        if any("Yer Değiştirme (u)" in res for res in load_results):
            x_vals = np.arange(3)
            plots = [res["Yer Değiştirme (u)"] for res in load_results if "Yer Değiştirme (u)" in res]
            labels = [f"Yük Senaryosu {res['Yük Senaryosu']}" for res in load_results if "Yer Değiştirme (u)" in res]
            self.plot_graph(x_vals, plots, xlabel="Düğüm", ylabel="Yer Değiştirme (u)", title="Yük Senaryoları",
                           labels=labels, figsize=(4, 3))