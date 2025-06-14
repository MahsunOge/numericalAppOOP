# ode.py
import matplotlib.pyplot as plt
from base_module import BaseModule
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import time
from matplotlib import animation
from streamlit.components.v1 import html
import io

class ODEModule(BaseModule):
    def __init__(self):
        super().__init__()
        if 'ode_solutions' not in self.session_state:
            self.session_state.ode_solutions = {}
        if 'pendulum_solution' not in self.session_state:
            self.session_state.pendulum_solution = None



    def run(self):
        st.header("Adi Diferansiyel Denklem (ODE) Çözücüleri")
        st.markdown("""
        - **RK45**: Uyarlanabilir, genel amaçlı.
        - **RK23**: Daha az hassas, hızlı.
        - **DOP853**: Yüksek hassasiyet.
        - **LSODA**: Sert problemler için uygun.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("ODE Tanımı")
            ode_func_str = st.text_input("dy/dt = f(t, y):", "-0.5 * y", key="ode_func_str")
            y0_str = st.text_input("Başlangıç Koşulları y(t0):", "1.0", key="ode_y0")
            t_start = st.number_input("Başlangıç Zamanı (t0):", 0.0, key="ode_t_start")
            t_end = st.number_input("Bitiş Zamanı (t_end):", 10.0, key="ode_t_end")
            num_t_points = st.number_input(label="Zaman Noktası Sayısı:", value=100, min_value=10, key="ode_num_t_points")
            try:
                y0 = np.array([float(val.strip()) for val in y0_str.split(',')])
            except ValueError:
                st.error("Geçerli sayısal başlangıç koşulları girin.")
                return

            def ode_system(t, y_vec):
                try:
                    local_vars = {'t': t, 'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'sqrt': np.sqrt}
                    local_vars['y'] = y_vec[0] if len(y_vec) == 1 else y_vec
                    odes = [s.strip() for s in ode_func_str.split(',')]
                    if len(odes) != len(y_vec):
                        st.error("ODE denklem sayısı ile başlangıç koşulları uyuşmuyor.")
                        return np.zeros_like(y_vec)
                    return np.array([eval(s, {"__builtins__": {}}, local_vars) for s in odes])
                except Exception as e:
                    st.error(f"ODE fonksiyon hatası: {e}")
                    return np.zeros_like(y_vec)

            t_eval = np.linspace(t_start, t_end, num_t_points)
            solvers = ['RK45', 'RK23', 'DOP853', 'LSODA']

            if st.button("ODE Sistemini Çöz"):
                self.session_state.ode_solutions = {}
                for solver in solvers:
                    try:
                        start_time = time.perf_counter()
                        sol = solve_ivp(ode_system, [t_start, t_end], y0, method=solver, t_eval=t_eval, dense_output=True)
                        time_taken = (time.perf_counter() - start_time) * 1000
                        if sol.success:
                            self.session_state.ode_solutions[solver] = {'sol': sol, 'time': time_taken}
                        else:
                            st.warning(f"{solver} başarısız: {sol.message}")
                    except Exception as e:
                        st.error(f"{solver} hatası: {e}")

        if self.session_state.ode_solutions:
            with col2:
                st.subheader("Çözüm Grafikleri")
                plots = []
                labels = []
                for i in range(len(y0)):
                    for name, data in self.session_state.ode_solutions.items():
                        plots.append(data['sol'].y[i])
                        labels.append(f'{name} - y{i} (Süre: {data["time"]:.2f}ms)')
                self.plot_graph(t_eval, plots, xlabel="Zaman (t)", ylabel="y(t)", title="ODE Çözümleri", labels=labels, figsize=(6, 4))

        self.pendulum_simulation()

    def pendulum_simulation(self):
        st.subheader("Uygulama: Basit Sarkaç Simülasyonu")
        length = st.slider("Sarkaç Uzunluğu L (m):", 0.1, 5.0, 1.0, key="pend_L")
        mass = st.slider("Kütle m (kg):", 0.1, 2.0, 0.5, key="pend_m")
        damping_b = st.slider("Sönümleme Katsayısı b (kg/s):", 0.0, 1.0, 0.1, key="pend_b")
        gravity_g = st.number_input("Yerçekimi İvmesi g (m/s²):", value=9.81, key="pend_g")
        theta0_deg = st.slider("Başlangıç Açısı θ₀ (derece):", -90.0, 90.0, 30.0, key="pend_theta0_deg")
        omega0 = st.slider("Başlangıç Açısal Hızı ω₀ (rad/s):", -5.0, 5.0, 0.0, key="pend_omega0")
        t_sim_end = st.slider("Simülasyon Süresi (s):", 1.0, 30.0, 10.0, key="pend_t_sim_end")

        theta0_rad = np.deg2rad(theta0_deg)
        y0_pendulum = [theta0_rad, omega0]
        pendulum_t_eval = np.linspace(0, t_sim_end, 200)

        def pendulum_system(t, y_pend):
            theta, omega = y_pend
            dtheta_dt = omega
            domega_dt = -(gravity_g / length) * np.sin(theta) - (damping_b / mass) * omega
            return [dtheta_dt, domega_dt]

        if st.button("Sarkaç Simülasyonunu Başlat"):
            sol_pendulum = solve_ivp(pendulum_system, [0, t_sim_end], y0_pendulum,
                                    method='RK45', t_eval=pendulum_t_eval, dense_output=True)
            self.session_state.pendulum_solution = sol_pendulum

        if self.session_state.pendulum_solution:
            sol = self.session_state.pendulum_solution
            fig, (ax_angle, ax_phase) = plt.subplots(1, 2, figsize=(12, 5))
            ax_angle.plot(sol.t, np.rad2deg(sol.y[0]), label='Açı θ(t) (derece)')
            ax_angle.plot(sol.t, sol.y[1], label='Açısal Hız ω(t) (rad/s)')
            ax_angle.set_xlabel("Zaman (s)")
            ax_angle.set_ylabel("Değer")
            ax_angle.legend()
            ax_angle.grid(True)
            ax_angle.set_title("Sarkaç Açı ve Hız")

            ax_phase.plot(np.rad2deg(sol.y[0]), sol.y[1])
            ax_phase.set_xlabel("Açı θ (derece)")
            ax_phase.set_ylabel("Açısal Hız ω (rad/s)")
            ax_phase.grid(True)
            ax_phase.set_title("Faz Portresi")
            st.pyplot(fig)
            plt.close(fig)

            st.write("Sarkaç Animasyonu")
            animation_placeholder = st.empty()
            animation_placeholder.text("Animasyon oluşturuluyor...")

            fig_anim, ax_anim = plt.subplots(figsize=(6, 4))
            ax_anim.set_xlim([-length*1.2, length*1.2])
            ax_anim.set_ylim([-length*1.2, length*0.2])
            ax_anim.set_aspect('equal')
            ax_anim.grid(True)
            line_anim, = ax_anim.plot([], [], 'o-', lw=2, markersize=8)
            time_text_anim = ax_anim.text(0.05, 0.9, '', transform=ax_anim.transAxes)

            def init_anim():
                line_anim.set_data([], [])
                time_text_anim.set_text('')
                return line_anim, time_text_anim

            def animate_pendulum(i):
                theta_i = sol.y[0][i]
                x_bob = length * np.sin(theta_i)
                y_bob = -length * np.cos(theta_i)
                line_anim.set_data([0, x_bob], [0, y_bob])
                time_text_anim.set_text(f'Zaman = {sol.t[i]:.1f}s')
                return line_anim, time_text_anim

            num_frames = min(100, len(sol.t))
            frame_indices = np.linspace(0, len(sol.t) - 1, num_frames, dtype=int)
            html_buffer = self.save_animation(
                fig_anim,
                lambda i: animate_pendulum(frame_indices[i]),
                frames=num_frames,
                init_func=init_anim,
                interval=max(20, int(t_sim_end * 1000 / num_frames)),
                height=500
            )
            if html_buffer:
                animation_placeholder.empty()  # "Animasyon oluşturuluyor..." yazısını kaldır
            plt.close(fig_anim)