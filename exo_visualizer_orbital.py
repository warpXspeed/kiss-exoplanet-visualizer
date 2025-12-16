import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="KISS Exoplanet Visualizer", layout="wide")
st.title("🌌 KISS Exoplanet Visualizer")
st.markdown("**Beautiful orbital architectures** — known planets in **cyan**, predictions in **magenta**. Now with 2D top-down orbital overview!")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('filled_systems.csv')

df = load_data()

# Sidebar
st.sidebar.header("Controls")
systems = sorted(df['System'].unique())
selected = st.sidebar.selectbox("Select System", systems)

show_hz = st.sidebar.checkbox("Show Approximate Habitable Zone", value=True)
dark_mode = st.sidebar.checkbox("Dark Space Theme", value=True)
show_2d = st.sidebar.checkbox("Show 2D Orbital Overview", value=True)

# Get row
row = df[df['System'] == selected].iloc[0]

def parse_planets(string):
    if pd.isna(string) or string == 'None' or string.strip() == '':
        return [], []
    planets = [p.strip() for p in string.split(';')]
    a_list, labels = [], []
    for p in planets:
        try:
            per_str, a_str = p.split(' (')
            a = float(a_str.rstrip(')'))
            a_list.append(a)
            labels.append(per_str + "d")
        except:
            continue
    return a_list, labels

known_a, known_labels = parse_planets(row['Known_Planets_Pdays_aAU'])
pred_a, pred_labels = parse_planets(row['Predicted_Unseen_Pdays_aAU'])

all_a = known_a + pred_a
if not all_a:
    st.write("No orbital data for this system.")
    st.stop()

# === 1D Linear View ===
fig1, ax1 = plt.subplots(figsize=(14, 6))
if dark_mode:
    plt.style.use('dark_background')
    fig1.patch.set_facecolor('black')
    ax_bg = '#0e1117'
else:
    ax_bg = 'white'
ax1.set_facecolor(ax_bg)

ax1.scatter(0.01, 0, s=800, color='gold', marker='*', edgecolors='orange', linewidth=2, zorder=10)

y_known = np.ones(len(known_a)) * 1.0
ax1.scatter(known_a, y_known, s=150, color='cyan', edgecolors='white', linewidth=1,
                 label=f'Known ({len(known_a)})', zorder=5)
for a, lab in zip(known_a, known_labels):
    ax1.text(a, 1.15, lab, ha='center', va='bottom', fontsize=10, color='cyan', fontweight='bold')

if pred_a:
    y_pred = np.ones(len(pred_a)) * 0.85
    ax1.scatter(pred_a, y_pred, s=250, color='magenta', marker='*', edgecolors='white',
                 label=f'Predicted ({len(pred_a)})', zorder=5)
    for a, lab in zip(pred_a, pred_labels):
        ax1.text(a, 0.7, lab + " (pred)", ha='center', va='top', fontsize=10, color='magenta', fontweight='bold')

all_sorted_a = sorted(all_a)
for i in range(len(all_sorted_a)-1):
    ax1.plot(all_sorted_a[i:i+2], [1.0, 1.0], color='gray', ls='--', alpha=0.5, lw=1)

if show_hz and row['Stellar_Mass_Msun'] > 0:
    lum_ratio = (row['Stellar_Mass_Msun']) ** 3.5
    hz_inner = 0.8 / np.sqrt(lum_ratio)
    hz_outer = 1.8 / np.sqrt(lum_ratio)
    ax1.axvspan(hz_inner, hz_outer, alpha=0.15, color='green', label='Approx Habitable Zone')

ax1.set_xscale('log')
min_a = min(all_a) * 0.7
max_a = max(all_a) * 1.5
ax1.set_xlim(max(0.005, min_a), max_a)
ax1.set_ylim(0.5, 1.5)
ax1.set_yticks([])
ax1.set_xlabel("Semi-Major Axis (AU) — Logarithmic Scale", fontsize=12)
ax1.set_title(f"{selected} • {row['Num_Known']} Known • {row['Stellar_Mass_Msun']} M⊙", fontsize=14)
ax1.legend(loc='upper left', frameon=True, fancybox=True,
           facecolor='black' if dark_mode else 'white')
ax1.grid(True, which='both', ls=':', alpha=0.3, color='gray')

st.pyplot(fig1)

# === 2D Orbital Overview ===
if show_2d:
    fig2 = plt.figure(figsize=(10, 10))
    if dark_mode:
        fig2.patch.set_facecolor('black')
        ax_bg = '#0e1117'
    else:
        ax_bg = 'white'
    ax2 = fig2.add_subplot(111, projection='polar')
    ax2.set_facecolor(ax_bg)

    # Star
    ax2.scatter(0, 0, s=1000, color='gold', marker='*', edgecolors='orange', linewidth=2, zorder=10)

    # sqrt transform for better inner planet visibility
    def to_plot_r(a): return np.sqrt(a)
    max_a = max(all_a) * 1.2
    max_pr = to_plot_r(max_a)

    # Nice grid circles (individual orbits + reference AU lines)
    nice_au = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    nice_au = nice_au[nice_au <= max_a * 1.5]
    grid_au = np.unique(np.concatenate([all_a, nice_au]))

    theta_full = np.linspace(0, 2*np.pi, 200)
    for a in grid_au:
        pr = to_plot_r(a)
        ax2.plot(theta_full, np.full_like(theta_full, pr), color='gray', ls=':', alpha=0.3, lw=1)

    # Known planets
    for a, lab in zip(known_a, known_labels):
        theta_pos = np.random.uniform(0, 2*np.pi)
        pr = to_plot_r(a)
        ax2.scatter(theta_pos, pr, s=150, color='cyan', edgecolors='white', zorder=5)
        ax2.text(theta_pos, pr * 1.08, lab, ha='center', va='center', fontsize=9, color='cyan',
                 rotation=theta_pos*180/np.pi - 90, rotation_mode='anchor')

    # Predicted planets
    for a, lab in zip(pred_a, pred_labels):
        theta_pos = np.random.uniform(0, 2*np.pi)
        pr = to_plot_r(a)
        ax2.scatter(theta_pos, pr, s=250, color='magenta', marker='*', edgecolors='white', zorder=5)
        ax2.text(theta_pos, pr * 0.92, lab + " (pred)", ha='center', va='center', fontsize=9, color='magenta',
                 rotation=theta_pos*180/np.pi - 90, rotation_mode='anchor')

    # Habitable zone
    if show_hz and row['Stellar_Mass_Msun'] > 0:
        lum_ratio = (row['Stellar_Mass_Msun']) ** 3.5
        hz_inner = max(0.8 / np.sqrt(lum_ratio), min(all_a)*0.5)
        hz_outer = min(1.8 / np.sqrt(lum_ratio), max(all_a)*1.5)
        theta = np.linspace(0, 2*np.pi, 200)
        ax2.fill_between(theta, to_plot_r(hz_inner), to_plot_r(hz_outer),
                         color='green', alpha=0.15, label='Approx Habitable Zone')

    # Custom radial ticks with real AU labels
    tick_au = nice_au
    ax2.set_rticks(to_plot_r(tick_au))
    tick_labels = [f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in tick_au]
    ax2.set_yticklabels(tick_labels)

    ax2.set_rlim(0, max_pr)
    ax2.set_thetagrids([])  # hide angle ticks
    ax2.set_title("2D Orbital Overview (Top-Down)\nSquare-Root Radial Scaling", fontsize=14, pad=30)
    ax2.grid(True, ls=':', alpha=0.4)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, fancybox=True,
               facecolor='black' if dark_mode else 'white')

    st.pyplot(fig2)

# Info section
st.markdown(f"""
**System Highlights**  
- **Known Planets**: {row['Num_Known']}  
- **Predicted Unseen**: {len(pred_a)}  
- **Stellar Mass**: {row['Stellar_Mass_Msun']} M⊙  
""")

st.caption("KISS Principle: Elegant geometric spacing → natural harmony. 2D view assumes circular orbits.")
