"""Generate diagram comparing No AO, Single-Conjugate AO, and MCAO."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = ['Nanum Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_atmosphere_and_telescope(ax, title, dm_altitudes, show_correction=True):
    """Draw a single panel showing atmosphere layers, telescope, and DM correction.

    Args:
        ax: Matplotlib axes.
        title: Panel title.
        dm_altitudes: List of DM conjugation altitudes in km (empty = no AO).
        show_correction: Whether to show correction effect.
    """
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Turbulence layers with Cn2 fractions
    layers = [
        (0.0, 0.40, '지면층 / Ground', '#FF6B6B'),
        (1.0, 0.10, '경계층 / Boundary', '#FFA07A'),
        (3.0, 0.15, '자유대기 / Free atm.', '#FFD700'),
        (6.0, 0.10, '대류권 / Tropopause', '#87CEEB'),
        (10.0, 0.10, 'Jet stream', '#9370DB'),
    ]

    # Draw atmosphere layers
    for h, cn2, label, color in layers:
        # Turbulence strength visualization (wavy lines)
        x_wave = np.linspace(-3.5, 3.5, 200)
        amplitude = cn2 * 8
        n_waves = int(5 + cn2 * 30)
        y_wave = h + amplitude * np.sin(n_waves * np.pi * x_wave / 7)

        ax.fill_between(x_wave, h - 0.15, y_wave, alpha=0.15, color=color)
        ax.plot(x_wave, y_wave, color=color, lw=1.5, alpha=0.6)

        # Layer label
        ax.text(3.8, h, f'{label}\n{h:.0f} km ({cn2*100:.0f}%)',
                fontsize=6.5, va='center', ha='left', color='gray')

    # Telescope at bottom
    tel_y = -0.8
    tel = patches.FancyBboxPatch((-1, tel_y - 0.3), 2, 0.6,
                                  boxstyle='round,pad=0.1',
                                  facecolor='#2C3E50', edgecolor='black',
                                  linewidth=1.5)
    ax.add_patch(tel)
    ax.text(0, tel_y, 'NST 1.6m', fontsize=8, ha='center', va='center',
            color='white', fontweight='bold')

    # Light rays from top
    for x_start in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
        ax.plot([x_start, x_start * 0.3], [11.5, tel_y + 0.3],
                color='gold', lw=0.8, alpha=0.3)

    # DM positions and correction zones
    dm_colors = ['#2ECC71', '#3498DB', '#E74C3C']
    for i, dm_h in enumerate(dm_altitudes):
        color = dm_colors[i % len(dm_colors)]

        # DM marker
        dm_rect = patches.FancyBboxPatch((-3.2, dm_h - 0.15), 1.2, 0.3,
                                          boxstyle='round,pad=0.05',
                                          facecolor=color, edgecolor='black',
                                          linewidth=1.5, alpha=0.9)
        ax.add_patch(dm_rect)
        ax.text(-2.6, dm_h, f'DM #{i+1}', fontsize=7, ha='center',
                va='center', color='white', fontweight='bold')

        if show_correction:
            # Correction zone (connecting DM to the layer it corrects)
            correction_range = 2.0  # km effective range
            rect = patches.Rectangle((-3.5, dm_h - correction_range * 0.4),
                                      7, correction_range * 0.8,
                                      linewidth=1.5, edgecolor=color,
                                      facecolor=color, alpha=0.08,
                                      linestyle='--')
            ax.add_patch(rect)

            # Checkmark on corrected layers
            for h, cn2, label, lcolor in layers:
                if abs(h - dm_h) < correction_range:
                    ax.text(-3.8, h, '✓', fontsize=12, color=color,
                            fontweight='bold', va='center')


def draw_fov_result(ax, title, strehl_map, label):
    """Draw FOV correction result as a 2D heatmap.

    Args:
        ax: Matplotlib axes.
        title: Panel title.
        strehl_map: 2D array of Strehl values.
        label: Description text.
    """
    im = ax.imshow(strehl_map, cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[-35, 35, -35, 35], origin='lower')
    ax.set_xlabel('Arcsec', fontsize=9)
    ax.set_ylabel('Arcsec', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')

    # FOV border
    rect = patches.Rectangle((-35, -35), 70, 70, linewidth=2,
                              edgecolor='white', facecolor='none',
                              linestyle='--')
    ax.add_patch(rect)

    # Center cross
    ax.plot(0, 0, '+', color='white', markersize=10, markeredgewidth=1.5)

    # Label
    ax.text(0, -42, label, fontsize=8, ha='center', va='top',
            style='italic', color='#555')

    return im


def make_strehl_map(mode='none', size=100):
    """Generate a synthetic Strehl ratio map across the FOV.

    Args:
        mode: 'none', 'scao', 'glao', 'mcao2', 'mcao3'.
        size: Grid size.

    Returns:
        2D numpy array of Strehl values.
    """
    x = np.linspace(-35, 35, size)
    y = np.linspace(-35, 35, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    if mode == 'none':
        # No AO: uniformly poor
        strehl = np.full_like(R, 0.05) + 0.03 * np.random.randn(*R.shape)

    elif mode == 'scao':
        # Single-conjugate: good at center, drops off rapidly
        theta0 = 3.0  # isoplanatic angle in arcsec
        strehl = 0.85 * np.exp(-(R / theta0)**2) + 0.05

    elif mode == 'glao':
        # Ground-layer AO: moderate improvement everywhere
        strehl = 0.35 + 0.15 * np.exp(-(R / 15)**2)

    elif mode == 'mcao2':
        # 2-DM MCAO: good over wider field
        strehl = 0.55 + 0.20 * np.exp(-(R / 25)**2)

    elif mode == 'mcao3':
        # 3-DM MCAO: nearly uniform across FOV
        strehl = 0.65 + 0.15 * np.exp(-(R / 40)**2)

    return np.clip(strehl, 0, 1)


# === Create the figure ===
fig = plt.figure(figsize=(20, 16))

# Top row: 3 atmospheric diagrams
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)

draw_atmosphere_and_telescope(
    ax1, 'No AO\n(적응광학 없음)', dm_altitudes=[], show_correction=False)
draw_atmosphere_and_telescope(
    ax2, 'Single-Conjugate AO\n(단일 공역 AO — AO-308)', dm_altitudes=[0])
draw_atmosphere_and_telescope(
    ax3, '3-Mirror MCAO\n(3-거울 다중 공역 AO)', dm_altitudes=[0, 3, 6])

# Add "vs" arrows between panels
fig.text(0.355, 0.75, '→', fontsize=30, ha='center', va='center',
         color='gray', fontweight='bold')
fig.text(0.66, 0.75, '→', fontsize=30, ha='center', va='center',
         color='gray', fontweight='bold')

# Bottom row: FOV Strehl maps
ax4 = fig.add_subplot(2, 4, 5)
ax5 = fig.add_subplot(2, 4, 6)
ax6 = fig.add_subplot(2, 4, 7)
ax7 = fig.add_subplot(2, 4, 8)

np.random.seed(42)

im = draw_fov_result(ax4, 'No AO', make_strehl_map('none'),
                     '전체 시야 흐림\nEntire FOV blurred')
draw_fov_result(ax5, 'SCAO (1 DM)', make_strehl_map('scao'),
                f'중심 θ₀≈3" 만 선명\nOnly center θ₀≈3" sharp')
draw_fov_result(ax6, 'GLAO (ground)', make_strehl_map('glao'),
                '전체 약간 개선\nModest improvement everywhere')
draw_fov_result(ax7, 'MCAO (3 DM)', make_strehl_map('mcao3'),
                '전체 시야 균일 보정!\nUniform correction across FOV!')

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.35])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Strehl Ratio', fontsize=10)

# Main title
fig.suptitle(
    'Adaptive Optics Comparison: No AO → SCAO → MCAO\n'
    '적응광학 비교: AO 없음 → 단일 공역 AO → 다중 공역 AO',
    fontsize=16, fontweight='bold', y=0.98)

# Summary box
summary_text = (
    '핵심 / Key Point: 지면층 난류가 전체의 ~40% → '
    'Ground-layer DM이 가장 큰 효과 / Ground-layer DM has the largest effect\n'
    'SCAO: 중심 수 arcsec만 보정 / Corrects only center few arcsec   |   '
    'MCAO: 70"×70" 전체 시야 보정 목표 / Goal: correct entire 70"×70" FOV'
)
fig.text(0.5, 0.01, summary_text, fontsize=10, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='orange', alpha=0.9))

plt.subplots_adjust(left=0.05, right=0.90, top=0.92, bottom=0.08,
                    wspace=0.3, hspace=0.35)

output_path = '/Users/eunsupark/Library/CloudStorage/OneDrive-개인/Projects/StudyWithAI/Solar_Observation/papers/04_goode_cao_2012/ao_comparison_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to: {output_path}')
