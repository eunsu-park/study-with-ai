"""Generate diagrams comparing on-axis vs off-axis Gregorian telescope designs."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyArrowPatch, Arc

plt.rcParams['font.family'] = ['Nanum Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_on_axis_gregorian(ax):
    """Draw on-axis Gregorian telescope schematic."""
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('On-Axis Gregorian\n(일반 그레고리안)', fontsize=14, fontweight='bold',
                 pad=15)

    # Primary mirror (concave parabolic)
    theta = np.linspace(-0.4, 0.4, 100)
    pm_x = 8 + 0.8 * theta**2
    pm_y = theta * 6
    ax.plot(pm_x, pm_y, 'b-', linewidth=3, label='Primary (주경)')
    ax.text(9.2, 0, 'Primary\n(주경)\nParabolic\n(포물면)', fontsize=8,
            ha='left', va='center', color='blue')

    # Secondary mirror (small convex ellipsoidal) - IN THE WAY
    sec_x = 4.5
    theta_s = np.linspace(-0.12, 0.12, 50)
    sec_mx = sec_x - 0.3 * theta_s**2
    sec_my = theta_s * 6
    ax.plot(sec_mx, sec_my, 'r-', linewidth=3, label='Secondary (부경)')
    ax.text(4.5, -1.3, 'Secondary\n(부경)', fontsize=8, ha='center',
            va='top', color='red')

    # Incoming light rays
    for y_pos in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        # Ray to primary
        pm_hit_x = 8 + 0.8 * (y_pos/6)**2
        ax.annotate('', xy=(pm_hit_x, y_pos), xytext=(0, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gold',
                                   lw=1.5, alpha=0.7))

    # Reflected rays from primary to secondary
    for y_pos in [-2.0, -1.0, 1.0, 2.0]:
        pm_hit_x = 8 + 0.8 * (y_pos/6)**2
        ax.plot([pm_hit_x, sec_x], [y_pos, y_pos * 0.25], 'gold',
                lw=1.5, alpha=0.7)

    # Central ray blocked!
    pm_hit_x = 8 + 0.8 * (0/6)**2
    ax.plot([pm_hit_x, sec_x], [0, 0], 'gold', lw=1.5, alpha=0.7)

    # Blocked region (central obscuration)
    blocked = patches.Rectangle((0, -0.7), 4.5, 1.4, linewidth=0,
                                 facecolor='red', alpha=0.12)
    ax.add_patch(blocked)
    ax.text(2.2, 0.95, '⚠ Central Obscuration\n(중앙 차폐 영역)',
            fontsize=9, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='red', alpha=0.9))

    # Shadow lines from secondary
    ax.plot([0, sec_x], [0.7, 0.7], 'r--', lw=1, alpha=0.5)
    ax.plot([0, sec_x], [-0.7, -0.7], 'r--', lw=1, alpha=0.5)

    # Reflected rays from secondary to final focus
    for y_pos in [-2.0, -1.0, 1.0, 2.0]:
        ax.plot([sec_x, 10], [y_pos * 0.25, 0], 'orange', lw=1.2, alpha=0.5)

    # Final focus
    ax.plot(10, 0, 'k*', markersize=12)
    ax.text(10, -0.5, 'Focus\n(초점)', fontsize=8, ha='center')

    # Prime focus label
    ax.plot(6.2, 0, 'kx', markersize=8)
    ax.text(6.2, 0.5, 'Prime focus\n(주초점)', fontsize=7, ha='center',
            color='gray')


def draw_off_axis_gregorian(ax):
    """Draw off-axis Gregorian telescope schematic (NST style)."""
    ax.set_xlim(-1, 11)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Off-Axis Gregorian (NST)\n(비축 그레고리안)', fontsize=14,
                 fontweight='bold', pad=15)

    # Full parent paraboloid (ghosted)
    theta = np.linspace(-0.4, 0.4, 100)
    pm_x = 8 + 0.8 * theta**2
    pm_y = theta * 6
    ax.plot(pm_x, pm_y, color='blue', linewidth=1, alpha=0.2, linestyle='--')
    ax.text(9.2, 2.0, '(Parent\nparaboloid)', fontsize=7, color='blue',
            alpha=0.4, ha='left')

    # Off-axis segment (only upper part used)
    theta_off = np.linspace(0.05, 0.35, 80)
    pm_off_x = 8 + 0.8 * theta_off**2
    pm_off_y = theta_off * 6
    ax.plot(pm_off_x, pm_off_y, 'b-', linewidth=4)
    ax.text(9.2, 1.0, 'Off-axis\nSegment\n(비축 구간)', fontsize=8,
            ha='left', va='center', color='blue')

    # Secondary mirror (positioned OFF-AXIS, below)
    sec_x = 5.0
    sec_y_center = -1.5
    theta_s = np.linspace(-0.08, 0.08, 50)
    sec_mx = sec_x - 0.4 * theta_s**2
    sec_my = sec_y_center + theta_s * 5
    ax.plot(sec_mx, sec_my, 'r-', linewidth=3)
    ax.text(5.0, -2.5, 'Secondary\n(부경)\nEllipsoidal\n(타원면)', fontsize=8,
            ha='center', va='top', color='red')

    # Incoming light rays (only hitting the off-axis segment)
    y_positions = [0.5, 1.0, 1.5, 2.0]
    for y_pos in y_positions:
        pm_hit_x = 8 + 0.8 * (y_pos/6)**2
        # Incoming ray
        ax.annotate('', xy=(pm_hit_x, y_pos), xytext=(0, y_pos),
                    arrowprops=dict(arrowstyle='->', color='gold',
                                   lw=1.8, alpha=0.8))

    # Reflected rays from primary to prime focus area, then to secondary
    prime_focus_x = 6.0
    prime_focus_y = -0.5
    for y_pos in y_positions:
        pm_hit_x = 8 + 0.8 * (y_pos/6)**2
        ax.plot([pm_hit_x, prime_focus_x], [y_pos, prime_focus_y],
                'orange', lw=1.5, alpha=0.7)

    # Prime focus + field stop
    ax.plot(prime_focus_x, prime_focus_y, 'kx', markersize=10)
    ax.text(6.0, 0.2, 'Prime focus\n+ Field stop\n(주초점 + 열 차단)',
            fontsize=7, ha='center', color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen',
                      alpha=0.3))

    # Rays from prime focus to secondary
    for dy in [-0.3, 0, 0.3]:
        ax.plot([prime_focus_x, sec_x], [prime_focus_y, sec_y_center + dy],
                'orange', lw=1.2, alpha=0.5)

    # Reflected from secondary to final focus
    final_x = 10
    final_y = -1.5
    for dy in [-0.3, 0, 0.3]:
        ax.plot([sec_x, final_x], [sec_y_center + dy, final_y],
                'darkorange', lw=1.2, alpha=0.5)

    # Final focus
    ax.plot(final_x, final_y, 'k*', markersize=12)
    ax.text(10, -2.2, 'Focus\n(초점)', fontsize=8, ha='center')

    # NO obscuration zone - highlight
    no_block = patches.FancyBboxPatch((0.5, 0.2), 4.0, 2.2,
                                      boxstyle='round,pad=0.2',
                                      linewidth=2, edgecolor='green',
                                      facecolor='green', alpha=0.08)
    ax.add_patch(no_block)
    ax.text(2.5, 1.3, '✓ No Obscuration!\n(차폐 없음!)',
            fontsize=10, ha='center', color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='green', alpha=0.9))

    # Optical axis (ghosted)
    ax.plot([0, 10], [0, 0], 'k:', lw=0.5, alpha=0.3)
    ax.text(0.5, -0.3, 'Optical axis\n(광축)', fontsize=6, color='gray',
            alpha=0.5)


def draw_psf_comparison(ax_psf, ax_mtf):
    """Draw PSF and MTF comparison between on-axis and off-axis."""
    from scipy.special import j1

    # PSF
    theta = np.linspace(-8, 8, 1000)
    theta_safe = np.where(np.abs(theta) < 1e-10, 1e-10, theta)

    # Unobscured (off-axis)
    airy = (2 * j1(theta_safe) / theta_safe) ** 2

    # With 33% obscuration (typical on-axis)
    eps = 0.33
    airy_outer = 2 * j1(theta_safe) / theta_safe
    airy_inner = 2 * j1(eps * theta_safe) / (eps * theta_safe)
    psf_obs = ((airy_outer - eps**2 * airy_inner) / (1 - eps**2)) ** 2

    ax_psf.semilogy(theta / np.pi, airy, 'g-', lw=2,
                    label='Off-axis (ε=0, NST)')
    ax_psf.semilogy(theta / np.pi, psf_obs, 'r--', lw=2,
                    label='On-axis (ε=0.33)')
    ax_psf.set_xlabel('Angle (λ/D units)', fontsize=10)
    ax_psf.set_ylabel('Normalized Intensity', fontsize=10)
    ax_psf.set_title('PSF Comparison / PSF 비교', fontsize=12,
                     fontweight='bold')
    ax_psf.set_ylim(1e-4, 1.5)
    ax_psf.legend(fontsize=9)
    ax_psf.grid(True, alpha=0.3)

    # Highlight the difference region
    mask = (np.abs(theta/np.pi) > 0.5) & (np.abs(theta/np.pi) < 2.5)
    ax_psf.fill_between(theta[mask]/np.pi, airy[mask], psf_obs[mask],
                        alpha=0.15, color='red',
                        label='Contrast loss\n(대비 손실)')
    ax_psf.annotate('More stray light\n(산란광 증가)',
                    xy=(1.2, 0.005), fontsize=8, color='red',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    xytext=(1.8, 0.05))

    # MTF
    freq = np.linspace(0, 1, 500)

    # Unobscured OTF
    f = np.clip(freq, 0, 0.9999)
    otf_full = (2/np.pi) * (np.arccos(f) - f * np.sqrt(1 - f**2))

    # Obscured OTF (approximate)
    f_inner = np.clip(freq / eps, 0, 0.9999)
    otf_inner = (2/np.pi) * (np.arccos(f_inner) - f_inner * np.sqrt(1 - f_inner**2))
    otf_obs = np.clip((otf_full - eps**2 * otf_inner) / (1 - eps**2), 0, 1)

    ax_mtf.plot(freq, otf_full, 'g-', lw=2, label='Off-axis (ε=0, NST)')
    ax_mtf.plot(freq, otf_obs, 'r--', lw=2, label='On-axis (ε=0.33)')
    ax_mtf.set_xlabel('Normalized Spatial Frequency\n($f / f_{cutoff}$)',
                      fontsize=10)
    ax_mtf.set_ylabel('MTF', fontsize=10)
    ax_mtf.set_title('MTF Comparison / MTF 비교', fontsize=12,
                     fontweight='bold')
    ax_mtf.legend(fontsize=9)
    ax_mtf.grid(True, alpha=0.3)

    # Highlight mid-frequency dip
    ax_mtf.fill_between(freq, otf_obs, otf_full, alpha=0.15, color='red')
    ax_mtf.annotate('Mid-freq contrast loss\n(중간 주파수 대비 손실)',
                    xy=(0.45, 0.25), fontsize=8, color='red', ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    xytext=(0.65, 0.5))


# Create the full figure
fig = plt.figure(figsize=(16, 14))

# Top row: telescope schematics
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
draw_on_axis_gregorian(ax1)
draw_off_axis_gregorian(ax2)

# Bottom row: PSF and MTF
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
draw_psf_comparison(ax3, ax4)

fig.suptitle('Off-Axis Gregorian Design: Why It Matters for Solar Telescopes\n'
             'Off-Axis 그레고리안 설계: 태양 망원경에서 중요한 이유',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_path = '/Users/eunsupark/Library/CloudStorage/OneDrive-개인/Projects/StudyWithAI/Solar_Observation/papers/04_goode_cao_2012/off_axis_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f'Saved to: {output_path}')
