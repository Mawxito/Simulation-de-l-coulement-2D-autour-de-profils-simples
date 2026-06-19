"""
MODULE D'AFFICHAGE - PROFIL NACA
Visualisation des champs, courbes de portance (Cl/Cd), Cp et zone de séparation.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def afficher_validation_cl_cd(angles_sim, Cl_sim, Cd_sim, ANGLES_REF, CL_EXP):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(ANGLES_REF, CL_EXP, 'o--', color='#1D9E75', lw=2, ms=8, label='Cl exp. (NACA)')
    ax.plot(angles_sim, Cl_sim, 's-', color='#E24B4A', lw=2, ms=8, label='Cl simulé (N.-S.)')
    for a, cl in zip(angles_sim, Cl_sim):
        ax.annotate(f'{cl:.2f}', (a, cl), xytext=(5, 6), textcoords='offset points', fontsize=9, color='#E24B4A')
    ax.axvline(x=14.5, color='gray', ls=':', lw=1.5)
    ax.text(14.7, 0.1, 'Décrochage\n(~15°)', fontsize=9, color='gray')
    ax.set_xlabel('Angle d\'attaque α (°)'); ax.set_ylabel('Coefficient de portance Cl')
    ax.set_title('Validation Cl(α) — NACA 0012')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([-1, 18]); ax.set_ylim([-0.2, 1.7])

    ax = axes[1]
    ax.plot(angles_sim, Cd_sim, 's-', color='#378ADD', lw=2, ms=8, label='Cd simulé')
    for a, cd in zip(angles_sim, Cd_sim):
        ax.annotate(f'{cd:.3f}', (a, cd), xytext=(5, 6), textcoords='offset points', fontsize=9, color='#378ADD')
    ax.set_xlabel('Angle d\'attaque α (°)'); ax.set_ylabel('Coefficient de traînée Cd')
    ax.set_title('Coefficient de traînée Cd(α)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([-1, 18])

    plt.suptitle('Validation aérodynamique — NACA 0012')
    plt.tight_layout()
    plt.savefig("NACA_Validation_Cl_Cd.png", dpi=150)
    plt.show()

def afficher_champs_vitesse_pression(ANGLES_REF, champs, resultats, X, Y, Lx, Ly):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for col, angle in enumerate(ANGLES_REF):
        u, v, p, masque = champs[angle]
        r = next(rr for rr in resultats if rr['angle'] == angle)
        vitesse = np.sqrt(u**2 + v**2)

        im1 = axes[0, col].imshow(vitesse, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=15)
        axes[0, col].contour(X, Y, masque, levels=[0.5], colors='black', linewidths=2)
        axes[0, col].quiver(X[::8, ::8], Y[::8, ::8], u[::8, ::8], v[::8, ::8], color='white', alpha=0.5, scale=300, headwidth=4)
        axes[0, col].set_title(f'Vitesse — α = {angle}°\nCl = {r["Cl"]:.2f}')
        if col == 0: axes[0, col].set_ylabel('Y (m)')

        im2 = axes[1, col].imshow(p, cmap='coolwarm', origin='lower', extent=[0, Lx, 0, Ly])
        axes[1, col].contour(X, Y, masque, levels=[0.5], colors='black', linewidths=2)
        axes[1, col].set_title(f'Pression — α = {angle}°')
        if col == 0: axes[1, col].set_ylabel('Y (m)')

    plt.suptitle('Champs de vitesse et pression — NACA 0012')
    plt.tight_layout()
    plt.savefig("NACA_Champs_4angles.png", dpi=150)
    plt.show()

def afficher_separation(ANGLES_REF, champs, resultats, X, Y, Lx, Ly, x_sep_vals):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for idx, angle in enumerate(ANGLES_REF):
        u, v, p, masque = champs[angle]
        ax = axes[idx // 2, idx % 2]
        im = ax.imshow(np.sqrt(u**2 + v**2), cmap='jet', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=15)
        ax.contour(X, Y, masque, levels=[0.5], colors='black', linewidths=2)
        
        u_ext = u.copy(); u_ext[masque] = 0.0
        ax.contourf(X, Y, u_ext, levels=[-100, 0], colors=['red'], alpha=0.35)
        patch = mpatches.Patch(color='red', alpha=0.35, label='Zone u < 0 (décollement)')
        
        r = next(rr for rr in resultats if rr['angle'] == angle)
        x_sep = x_sep_vals[idx]
        titre = f"α = {angle}° | Cl = {r['Cl']:.2f} | " + (f"Décollement x/c = {x_sep:.2f}" if x_sep < 1.0 else "Attaché")
        ax.set_title(titre)
        ax.legend(handles=[patch], loc='upper right')

    plt.suptitle('Zone de séparation (u < 0)')
    plt.tight_layout()
    plt.savefig("NACA_Zones_Separation.png", dpi=150)
    plt.show()


def afficher_cp(ANGLES_REF, champs, cfd_module, x_sep_vals, X, Y):
    """
    Trace la distribution du coefficient de pression Cp(x/c)
    sur l'extrados et l'intrados pour chaque angle d'attaque.
    Convention aéronautique : axe Cp inversé (Cp négatif en haut).
    """
    colors_up = ['#E24B4A', '#378ADD', '#1D9E75', '#F5A623']
    colors_lo = ['#FF9999', '#99C4F0', '#80D4B0', '#FAD08A']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for idx, angle in enumerate(ANGLES_REF):
        u, v, p, masque = champs[angle]
        ax = axes[idx // 2, idx % 2]

        x_up, cp_up, x_lo, cp_lo = cfd_module.calculer_cp(
            p, masque, X, Y, cfd_module.U_in, y_c=1.0
        )

        # Normaliser x par la corde C (x → x/c relatif au bord d'attaque)
        if len(x_up) > 0:
            x_le = min(x_up.min(), x_lo.min()) if len(x_lo) > 0 else x_up.min()
            x_up_norm = (x_up - x_le) / cfd_module.C
            x_lo_norm = (x_lo - x_le) / cfd_module.C if len(x_lo) > 0 else np.array([])

            ax.plot(x_up_norm, cp_up, color=colors_up[idx], lw=1.8, label='Extrados')
            if len(x_lo_norm) > 0:
                ax.plot(x_lo_norm, cp_lo, color=colors_lo[idx], lw=1.8, ls='--', label='Intrados')

        # Convention aéronautique : Cp négatif vers le haut
        ax.invert_yaxis()
        ax.axhline(0, color='gray', lw=0.8, ls=':')

        x_sep = x_sep_vals[idx]
        if x_sep < 1.0:
            ax.axvline(x_sep, color='red', lw=1.5, ls='--', alpha=0.7,
                       label=f'Décollement x/c={x_sep:.2f}')

        ax.set_xlabel('x/c'); ax.set_ylabel('Cp')
        ax.set_title(f'Distribution Cp — α = {angle}°')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

    plt.suptitle('Coefficient de pression Cp(x/c) — NACA 0012', fontsize=13)
    plt.tight_layout()
    plt.savefig("NACA_Distribution_Cp.png", dpi=150, bbox_inches='tight')
    plt.show()