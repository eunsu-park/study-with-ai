"""Diagrams for understanding magnetospheric current systems and magnetic storms.

Generates publication-quality figures explaining:
1. Magnetosphere cross-section with all current systems
2. Magnetic storm phases (Dst timeline)
3. Birkeland current circuit (3D perspective)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc, Circle, Ellipse
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

# Set Korean font
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False


def fig1_magnetosphere_currents():
    """Cross-section of Earth's magnetosphere with all major current systems."""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(-22, 14)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")

    # --- Earth ---
    earth = Circle((0, 0), 0.8, color="steelblue", zorder=10)
    ax.add_patch(earth)
    ax.text(0, 0, "Earth", ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=11)

    # --- Magnetopause boundary ---
    # Dayside: compressed, Nightside: stretched
    theta_day = np.linspace(-np.pi/2, np.pi/2, 100)
    r_day = 10
    x_day = r_day * np.cos(theta_day)
    y_day = r_day * np.sin(theta_day) * 0.7
    ax.plot(x_day, y_day, "gray", lw=2.5)

    # Tail boundary
    ax.plot([-20, 0], [7, r_day * 0.7 * np.sin(np.pi/2)], "gray", lw=2.5)
    ax.plot([-20, 0], [-7, -r_day * 0.7 * np.sin(np.pi/2)], "gray", lw=2.5)
    ax.text(8, 6.5, "Magnetopause\n(자기권계면)", fontsize=9, color="gray",
            fontstyle="italic")

    # --- Bow Shock ---
    theta_bs = np.linspace(-np.pi/2, np.pi/2, 100)
    r_bs = 13
    x_bs = r_bs * np.cos(theta_bs)
    y_bs = r_bs * np.sin(theta_bs) * 0.75
    ax.plot(x_bs, y_bs, "gray", lw=1.5, ls="--", alpha=0.5)
    ax.text(11.5, 8, "Bow Shock\n(충격파)", fontsize=8, color="gray", alpha=0.7)

    # --- Solar wind arrows ---
    for y_sw in np.arange(-6, 7, 1.5):
        ax.annotate("", xy=(12, y_sw), xytext=(14, y_sw),
                    arrowprops=dict(arrowstyle="->", color="gold", lw=1.5))
    ax.text(13.5, 7.5, "Solar Wind\n(태양풍) -->", fontsize=10, color="gold",
            fontweight="bold", ha="center")

    # --- Magnetic field lines ---
    for L in [2, 3, 4, 6]:
        lam = np.linspace(-np.pi/2.3, np.pi/2.3, 200)
        r = L * np.cos(lam)**2
        x = r * np.cos(lam)
        y = r * np.sin(lam)
        mask = r > 0.85
        ax.plot(x[mask], y[mask], "lightskyblue", lw=0.7, alpha=0.5)

    # Open field lines (tail)
    for y_start in [1.5, 2.5, 3.5, 5]:
        ax.plot([-20, -2], [y_start, y_start * 0.3], "lightskyblue", lw=0.7, alpha=0.4)
        ax.plot([-20, -2], [-y_start, -y_start * 0.3], "lightskyblue", lw=0.7, alpha=0.4)

    # ========== CURRENT SYSTEMS ==========

    # --- 1. Magnetopause Current (Chapman-Ferraro) ---
    # Flows dawn-to-dusk (out of page on top, into page on bottom)
    for y_mp in np.arange(-5, 6, 1.2):
        x_mp = np.sqrt(max(0, (r_day * 0.7)**2 * (1 - (y_mp / (r_day * 0.7))**2)))
        if x_mp > 1:
            ax.plot(x_mp * 1.0, y_mp, "o", color="magenta", ms=5, zorder=8)
    ax.annotate("① Magnetopause\n   Current\n   (자기권계면 전류)\n   $\\sim10^7$ A\n   dawn-to-dusk",
                xy=(9, 2), fontsize=9, color="magenta", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="magenta", alpha=0.9))

    # --- 2. Ring Current ---
    ring = Ellipse((0, 0), 8, 2.5, fill=False, color="red", lw=3,
                   ls="-", zorder=6)
    ax.add_patch(ring)
    # Arrows showing westward direction
    for angle in [30, 150, 210, 330]:
        rad = np.radians(angle)
        x_rc = 4 * np.cos(rad)
        y_rc = 1.25 * np.sin(rad)
        dx = -np.sin(rad) * 0.6
        dy = np.cos(rad) * 0.3
        ax.annotate("", xy=(x_rc + dx, y_rc + dy), xytext=(x_rc, y_rc),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.annotate("② Ring Current\n   (고리 전류)\n   $\\sim10^6$ A\n   westward",
                xy=(-4.5, -2.5), fontsize=9, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))

    # --- 3. Tail Current (Cross-tail) ---
    for x_tail in np.arange(-18, -4, 2):
        ax.annotate("", xy=(x_tail, -0.15), xytext=(x_tail, 0.15),
                    arrowprops=dict(arrowstyle="-", color="green", lw=0))
        ax.plot(x_tail, 0, "o", color="green", ms=6, zorder=8)
    ax.annotate("③ Tail Current\n   (꼬리 전류)\n   dawn-to-dusk",
                xy=(-16, 1.5), fontsize=9, color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9))

    # Neutral sheet
    ax.plot([-20, -2], [0, 0], "green", lw=1, ls=":", alpha=0.5)
    ax.text(-12, -0.7, "neutral sheet", fontsize=7, color="green", alpha=0.7)

    # --- 4. Birkeland Currents (Field-Aligned) ---
    # Downward into north pole
    ax.annotate("", xy=(0.3, 1.0), xytext=(1.5, 4),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))
    ax.annotate("", xy=(-0.3, 1.0), xytext=(-1.5, 4),
                arrowprops=dict(arrowstyle="<-", color="blue", lw=2.5, ls="--"))
    # South pole
    ax.annotate("", xy=(0.3, -1.0), xytext=(1.5, -4),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2.5))
    ax.annotate("", xy=(-0.3, -1.0), xytext=(-1.5, -4),
                arrowprops=dict(arrowstyle="<-", color="blue", lw=2.5, ls="--"))

    ax.annotate("④ Birkeland Currents\n   (자기장 정렬 전류)\n   $\\sim10^6$ A\n   space ↔ ionosphere",
                xy=(2, 5), fontsize=9, color="blue", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9))

    # --- 5. Electrojet ---
    ax.annotate("", xy=(1.2, 0.9), xytext=(-1.2, 0.9),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2.5))
    ax.annotate("", xy=(1.2, -0.9), xytext=(-1.2, -0.9),
                arrowprops=dict(arrowstyle="<->", color="orange", lw=2.5))
    ax.annotate("⑤ Electrojet\n   (전기분사류)\n   $\\sim10^5$ A\n   E-W in ionosphere",
                xy=(-5, 5.5), fontsize=9, color="orange", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.9))

    # --- Reconnection X-line ---
    ax.plot(-8, 0, "x", color="yellow", ms=15, mew=3, zorder=9)
    ax.text(-8, -1, "Reconnection\n(자기 재결합)", fontsize=8, color="yellow",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round", fc="black", ec="yellow", alpha=0.8))

    # --- Dayside reconnection ---
    ax.plot(9.5, 0, "x", color="yellow", ms=12, mew=3, zorder=9)
    ax.text(9.5, -1.3, "Dayside\nReconnection\n(B_z south일 때)",
            fontsize=7, color="yellow", ha="center",
            bbox=dict(boxstyle="round", fc="black", ec="yellow", alpha=0.7))

    ax.set_xlabel("Sunward ← → Tailward (R_E)", fontsize=12)
    ax.set_ylabel("North-South (R_E)", fontsize=12)
    ax.set_title("Earth's Magnetosphere: Major Current Systems\n"
                 "지구 자기권: 주요 전류 체계",
                 fontsize=15, fontweight="bold")
    ax.set_facecolor("#0a0a2a")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="magenta", marker="o", ms=6, lw=0,
               label="① Magnetopause current"),
        Line2D([0], [0], color="red", lw=3, label="② Ring current"),
        Line2D([0], [0], color="green", marker="o", ms=6, lw=0,
               label="③ Tail current"),
        Line2D([0], [0], color="blue", lw=2.5, label="④ Birkeland currents"),
        Line2D([0], [0], color="orange", lw=2.5, label="⑤ Electrojet"),
        Line2D([0], [0], color="yellow", marker="x", ms=10, lw=0, mew=2,
               label="Reconnection site"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              facecolor="#1a1a3a", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    plt.savefig("fig1_magnetosphere_currents.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: fig1_magnetosphere_currents.png")


def fig2_magnetic_storm_phases():
    """Magnetic storm phases shown through Dst index timeline."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    # --- Top: Dst timeline ---
    ax1 = axes[0]

    # Simulated Dst data for a typical storm
    t = np.linspace(-12, 120, 1000)  # hours

    # Build storm profile
    dst = np.zeros_like(t)
    # Quiet before storm
    dst[t < 0] = 5 * np.sin(t[t < 0] * 0.3)  # quiet day variation

    # SSC at t=0: sudden jump
    mask_ssc = (t >= 0) & (t < 1)
    dst[mask_ssc] = 30

    # Main phase: t=1 to t=12
    mask_main = (t >= 1) & (t < 12)
    dst[mask_main] = 30 - 180 * (1 - np.exp(-(t[mask_main] - 1) / 4))

    # Minimum at t~12
    dst_min = -150

    # Recovery phase: t=12 onwards (two-component decay)
    mask_rec = t >= 12
    t_rec = t[mask_rec] - 12
    dst[mask_rec] = dst_min * (0.4 * np.exp(-t_rec / 8) + 0.6 * np.exp(-t_rec / 60))

    ax1.plot(t, dst, "lime", lw=2.5, zorder=5)
    ax1.fill_between(t, dst, 0, where=(dst < 0), alpha=0.15, color="lime")
    ax1.axhline(0, color="white", lw=0.5, alpha=0.5)

    # Phase annotations
    # SSC
    ax1.axvspan(-2, 1, alpha=0.2, color="yellow")
    ax1.annotate("Phase 1\nSSC\n(초기 단계)\n\nCME 충격파\n자기권 압축\nDst ↑",
                 xy=(0, 30), xytext=(-8, 60),
                 fontsize=9, color="yellow", ha="center",
                 arrowprops=dict(arrowstyle="->", color="yellow"),
                 bbox=dict(boxstyle="round", fc="#1a1a3a", ec="yellow"))

    # Main phase
    ax1.axvspan(1, 12, alpha=0.15, color="red")
    ax1.annotate("Phase 2\nMain Phase\n(주 단계)\n\nRing current 강화\nDst 급락\n오로라 활발",
                 xy=(6, -100), xytext=(20, -40),
                 fontsize=9, color="red", ha="center",
                 arrowprops=dict(arrowstyle="->", color="red"),
                 bbox=dict(boxstyle="round", fc="#1a1a3a", ec="red"))

    # Recovery
    ax1.axvspan(12, 120, alpha=0.08, color="cyan")
    ax1.annotate("Phase 3\nRecovery\n(회복 단계)\n\nRing current 감소\n(charge exchange)\nDst → 0",
                 xy=(60, -40), xytext=(85, -100),
                 fontsize=9, color="cyan", ha="center",
                 arrowprops=dict(arrowstyle="->", color="cyan"),
                 bbox=dict(boxstyle="round", fc="#1a1a3a", ec="cyan"))

    # Storm classification lines
    for level, label, color in [(-50, "Moderate", "yellow"),
                                 (-100, "Intense", "orange"),
                                 (-250, "Super", "red")]:
        ax1.axhline(level, color=color, ls=":", lw=1, alpha=0.5)
        ax1.text(115, level + 5, label, fontsize=8, color=color, alpha=0.7)

    ax1.set_ylabel("Dst (nT)", fontsize=12, color="white")
    ax1.set_title("Magnetic Storm Phases — Dst Index\n"
                  "자기폭풍 단계 — Dst 지수", fontsize=14, fontweight="bold", color="white")
    ax1.set_xlim(-12, 120)
    ax1.set_ylim(-200, 80)
    ax1.grid(True, alpha=0.2, color="white")
    ax1.set_facecolor("#0a0a2a")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("white")
    ax1.spines["left"].set_color("white")
    ax1.spines["top"].set_color("#0a0a2a")
    ax1.spines["right"].set_color("#0a0a2a")

    # --- Bottom: Solar wind conditions (B_z and speed) ---
    ax2 = axes[1]

    # Simulated IMF B_z
    bz = np.zeros_like(t)
    bz[t < 0] = np.random.normal(0, 1, np.sum(t < 0))
    bz[(t >= 0) & (t < 1)] = 5  # shock compression: briefly northward
    mask_south = (t >= 1) & (t < 14)
    bz[mask_south] = -15 * np.exp(-(t[mask_south] - 1) / 8)  # strong southward
    bz[t >= 14] = np.random.normal(0, 2, np.sum(t >= 14))

    ax2.plot(t, bz, "white", lw=1.5, alpha=0.8)
    ax2.fill_between(t, bz, 0, where=(bz < 0), alpha=0.3, color="red",
                     label="Southward $B_z$ (reconnection!)")
    ax2.fill_between(t, bz, 0, where=(bz > 0), alpha=0.2, color="cyan",
                     label="Northward $B_z$ (shielded)")
    ax2.axhline(0, color="white", lw=0.5, alpha=0.5)

    ax2.annotate("Southward B_z → Reconnection\n→ Energy into magnetosphere\n→ STORM!",
                 xy=(5, -12), xytext=(30, -18),
                 fontsize=9, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"),
                 bbox=dict(boxstyle="round", fc="#1a1a3a", ec="red"))

    ax2.set_xlabel("Time (hours from CME arrival) / CME 도착 후 시간", fontsize=12, color="white")
    ax2.set_ylabel("IMF $B_z$ (nT)", fontsize=12, color="white")
    ax2.set_title("Interplanetary Magnetic Field — Key Driver\n"
                  "행성간 자기장 — 핵심 구동력", fontsize=12, color="white")
    ax2.set_xlim(-12, 120)
    ax2.legend(fontsize=9, facecolor="#1a1a3a", edgecolor="white", labelcolor="white")
    ax2.grid(True, alpha=0.2, color="white")
    ax2.set_facecolor("#0a0a2a")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("white")
    ax2.spines["left"].set_color("white")
    ax2.spines["top"].set_color("#0a0a2a")
    ax2.spines["right"].set_color("#0a0a2a")

    fig.set_facecolor("#0a0a2a")
    plt.tight_layout()
    plt.savefig("fig2_storm_phases.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved: fig2_storm_phases.png")


def fig3_current_circuit():
    """Birkeland's current circuit: 3D-like schematic showing closed loop."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.set_aspect("equal")

    # Title
    ax.set_title("Complete Current Circuit: Solar Wind → Magnetosphere → Ionosphere\n"
                 "완전한 전류 회로: 태양풍 → 자기권 → 전리층",
                 fontsize=14, fontweight="bold")

    # --- Boxes for each region ---
    # Solar Wind
    sw_box = patches.FancyBboxPatch((0, 6.5), 3, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#FFD700", edgecolor="black", lw=2, alpha=0.3)
    ax.add_patch(sw_box)
    ax.text(1.5, 7.25, "[Sun] Solar Wind\n태양풍\n(energy source)", ha="center", va="center",
            fontsize=10, fontweight="bold")

    # Magnetopause
    mp_box = patches.FancyBboxPatch((4, 6.5), 3, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#FF00FF", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(mp_box)
    ax.text(5.5, 7.25, "Magnetopause\n자기권계면\n(① magnetopause current)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Magnetotail
    mt_box = patches.FancyBboxPatch((8, 6.5), 2.5, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#00FF00", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(mt_box)
    ax.text(9.25, 7.25, "Magnetotail\n자기권 꼬리\n(③ tail current)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Reconnection
    recon_box = patches.FancyBboxPatch((7, 4), 3.5, 1.5, boxstyle="round,pad=0.2",
                                        facecolor="#FFFF00", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(recon_box)
    ax.text(8.75, 4.75, "[!] Reconnection\n자기 재결합\n(에너지 방출!)",
            ha="center", va="center", fontsize=10, fontweight="bold", color="darkred")

    # Ring Current
    rc_box = patches.FancyBboxPatch((4, 2), 3, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#FF0000", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(rc_box)
    ax.text(5.5, 2.75, "② Ring Current\n고리 전류\n(→ Dst index)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Birkeland Currents
    bc_box = patches.FancyBboxPatch((1, 2), 2.5, 1.5, boxstyle="round,pad=0.2",
                                     facecolor="#0000FF", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(bc_box)
    ax.text(2.25, 2.75, "④ Birkeland\nCurrents\n(field-aligned)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="blue")

    # Ionosphere / Electrojet
    ion_box = patches.FancyBboxPatch((0.5, -0.5), 4, 1.5, boxstyle="round,pad=0.2",
                                      facecolor="#FFA500", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(ion_box)
    ax.text(2.5, 0.25, "⑤ Ionosphere: Electrojet + Pedersen/Hall\n"
            "전리층: 전기분사류 + Pedersen/Hall 전류\n(→ AE index, 오로라!)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # --- Arrows connecting the circuit ---
    arrow_style = "Simple,tail_width=2,head_width=12,head_length=8"

    # Solar wind → Magnetopause
    ax.annotate("", xy=(4, 7.25), xytext=(3, 7.25),
                arrowprops=dict(arrowstyle="->", color="gold", lw=3))
    ax.text(3.5, 7.7, "driving\n구동", fontsize=8, ha="center", color="gold")

    # Magnetopause → Tail
    ax.annotate("", xy=(8, 7.25), xytext=(7, 7.25),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2.5))

    # Tail → Reconnection
    ax.annotate("", xy=(9.25, 5.5), xytext=(9.25, 6.5),
                arrowprops=dict(arrowstyle="->", color="green", lw=2.5))
    ax.text(9.8, 6, "에너지\n축적→방출", fontsize=8, ha="left", color="green")

    # Reconnection → Ring current
    ax.annotate("", xy=(7, 3), xytext=(8, 4),
                arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
    ax.text(7.8, 3.3, "입자 주입\ninjection", fontsize=8, color="red")

    # Ring current → Birkeland
    ax.annotate("", xy=(3.5, 2.75), xytext=(4, 2.75),
                arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))

    # Reconnection → Birkeland (direct)
    ax.annotate("", xy=(2.5, 3.5), xytext=(7, 4.5),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2.5, ls="--"))
    ax.text(4.5, 4.3, "direct driving\n직접 구동", fontsize=8, color="blue", ha="center")

    # Birkeland → Ionosphere
    ax.annotate("", xy=(2.25, 1), xytext=(2.25, 2),
                arrowprops=dict(arrowstyle="->", color="blue", lw=3))

    # Ionosphere → Birkeland (return)
    ax.annotate("", xy=(1, 2), xytext=(1, 1),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2, ls="--"))
    ax.text(0.3, 1.5, "return\n귀환", fontsize=8, color="blue")

    # --- Ground effects ---
    ground_box = patches.FancyBboxPatch((5.5, -0.5), 5, 1.5, boxstyle="round,pad=0.2",
                                         facecolor="#8B4513", edgecolor="black", lw=2, alpha=0.2)
    ax.add_patch(ground_box)
    ax.text(8, 0.25, "Ground Effects / 지상 영향\n"
            "GIC → 전력망 정전, 파이프라인 부식\n"
            "GPS 오차, 통신 장애, 위성 피해",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # Ionosphere → Ground
    ax.annotate("", xy=(5.5, 0.25), xytext=(4.5, 0.25),
                arrowprops=dict(arrowstyle="->", color="brown", lw=2.5))
    ax.text(5, 0.8, "dB/dt\n→ GIC", fontsize=8, color="brown", ha="center")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("fig3_current_circuit.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: fig3_current_circuit.png")


if __name__ == "__main__":
    fig1_magnetosphere_currents()
    fig2_magnetic_storm_phases()
    fig3_current_circuit()
    print("\nAll 3 figures saved!")
