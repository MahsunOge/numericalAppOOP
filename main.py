# main.py
import streamlit as st
from root_finding import RootFindingModule
from interpolation import InterpolationModule
from ode import ODEModule
from numerical_derivative import NumericalDerivativeModule
from numerical_integration import NumericalIntegrationModule
from linear_systems import LinearSystemsModule
from lu_decomposition import LUDecompositionModule
from optimization import OptimizationModule  # Yeni modül

class AppController:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="Sayısal Analiz Aracı")
        self.modules = {
            "Kök Bulma": RootFindingModule(),
            "İnterpolasyon": InterpolationModule(),
            "ODE Çözümü": ODEModule(),
            "Sayısal Türev": NumericalDerivativeModule(),
            "Sayısal İntegrasyon": NumericalIntegrationModule(),
            "Doğrusal Sistemler": LinearSystemsModule(),
            "LU Ayrıştırması": LUDecompositionModule(),
            "Optimizasyon": OptimizationModule(),  # Yeni modül
            # Diğer modüller buraya eklenecek
        }

    def run(self):
        st.sidebar.title("Sayısal Yöntemler")
        app_mode = st.sidebar.selectbox(
            "Bir Modül Seçin:",
            ["Giriş"] + list(self.modules.keys())
        )

        if app_mode == "Giriş":
            st.title("Sayısal Analiz Aracı ve Simülasyonlar")
            st.markdown("""
            Bu araç, sayısal analiz yöntemlerini interaktif bir şekilde keşfetmenizi sağlar.
            Sol taraftaki menüden bir yöntem seçerek başlayabilirsiniz.
            """)
        elif app_mode in self.modules:
            self.modules[app_mode].run()
        else:
            st.info(f"'{app_mode}' modülü henüz geliştirilmemiştir.")

if __name__ == "__main__":
    app = AppController()
    app.run()