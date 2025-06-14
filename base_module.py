# base_module.py
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io
from matplotlib import animation
from streamlit.components.v1 import html

class BaseModule:
    def __init__(self):
        self.session_state = st.session_state
        # FFmpeg yolunu ortam değişkenlerinden al veya varsayılan bir yol belirle
        self.ffmpeg_path = os.environ.get('FFMPEG_PATH', None)
        if self.ffmpeg_path:
            plt.rcParams['animation.ffmpeg_path'] = self.ffmpeg_path

    def parse_function_string(self, func_str):
        """
        String fonksiyonu ayrıştırır ve çalıştırılabilir bir fonksiyon ile SymPy ifadesine dönüştürür.

        Args:
            func_str (str): Kullanıcı tarafından girilen fonksiyon string'i (örneğin, 'sin(x)', 'x**2').

        Returns:
            tuple: (vectorized_f, expr)
                - vectorized_f: NumPy vektör işlemleri için uygun fonksiyon.
                - expr: SymPy sembolik ifadesi.
                - Hata durumunda (None, None) döner.
        """
        try:
            x = sp.symbols('x')
            expr = sp.sympify(func_str, locals={"sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                                               "sqrt": sp.sqrt, "log": sp.log, "tan": sp.tan})
            safe_globals = {
                "__builtins__": {},
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "log": np.log,
                "tan": np.tan
            }
            def f(x):
                local_vars = {"x": x}
                return eval(func_str, safe_globals, local_vars)
            vectorized_f = np.vectorize(f, otypes=[float])
            return vectorized_f, expr
        except Exception as e:
            st.error(f"Fonksiyon ayrıştırma hatası: {e}")
            return None, None

    def plot_graph(self, x, y_data, xlabel="", ylabel="", title="", labels=None, points=None, figsize=(6, 4)):
        """
        Grafik çizme metodu, tek veya çoklu veri serilerini destekler.

        Args:
            x (array-like): x ekseni verileri.
            y_data (array-like or list): y ekseni verileri (tek dizi veya dizi listesi).
            xlabel (str): x ekseni etiketi.
            ylabel (str): y ekseni etiketi.
            title (str): Grafik başlığı.
            labels (list): Veri serisi etiketleri.
            points (list): İşaretlenecek noktalar [(x, y, etiket), ...].
            figsize (tuple): Grafik boyutu (genişlik, yükseklik).

        Returns:
            None: Grafiği Streamlit'te gösterir.
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            if not isinstance(y_data, (list, np.ndarray)):
                y_data = [y_data]
            if isinstance(y_data[0], (list, np.ndarray)):
                for i, y in enumerate(y_data):
                    label = labels[i] if labels and i < len(labels) else f"Veri {i+1}"
                    ax.plot(x, y, label=label)
            else:
                label = labels[0] if labels else "Veri"
                ax.plot(x, y_data, label=label)
            if points:
                for point_x, point_y, point_label in points:
                    ax.scatter([point_x], [point_y], color='red', label=point_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Grafik çizme hatası: {e}")

    def save_animation(self, fig, animate_func, frames, init_func=None, interval=200, height=500, blit=True):
        """
        Matplotlib animasyonunu HTML formatında kaydeder ve Streamlit'te gösterir.

        Args:
            fig (matplotlib.figure.Figure): Matplotlib figürü.
            animate_func (callable): Her kareyi güncelleyen fonksiyon, Artist nesneleri listesi döndürmeli.
            frames (int or iterable): Frame sayısı veya frame dizisi.
            init_func (callable, optional): Animasyonun başlangıç durumunu ayarlayan fonksiyon.
            interval (float): Frame'ler arası gecikme (milisaniye).
            height (int): Streamlit HTML bileşeninin yüksekliği (piksel).
            blit (bool): Eğer True ise, sadece değişen kısımlar yeniden çizilir (Artist nesneleri gerekir).

        Returns:
            BytesIO: HTML içeriğini içeren BytesIO nesnesi, hata durumunda None.
        """
        try:
            # Animasyon oluştur
            ani = animation.FuncAnimation(
                fig=fig,
                func=animate_func,
                frames=frames,
                init_func=init_func,
                interval=interval,
                blit=blit
            )

            # HTML/JS formatına dönüştür
            html_content = ani.to_jshtml()

            # Streamlit'te göster
            html(html_content, height=height)

            # HTML içeriğini BytesIO'ya kaydet
            html_buffer = io.StringIO(html_content)
            html_bytes = io.BytesIO(html_buffer.getvalue().encode('utf-8'))
            html_buffer.close()
            html_bytes.seek(0)
            return html_bytes

        except Exception as e:
            st.error(f"Animasyon oluşturma hatası: {e}")
            return None