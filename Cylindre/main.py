"""
MODULE D'AFFICHAGE - GÉNÉRATION DES GRAPHIQUES MATPLOTLIB
Prend en charge l'exportation des figures pour les résultats de la simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse

def afficher_simulation_base(u, v, p, X, Y, x_c, y_c, D, Lx, Ly, U0, Re_base, iter_conv):
    """Affiche les champs de vitesse et de pression pour le régime stationnaire."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vitesse = np.sqrt(u**2 + v**2)
    
    im1 = axes[0].imshow(vitesse, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=U0*1.5)
    axes[0].quiver(X[::8, ::8], Y[::8, ::8], u[::8, ::8], v[::8, ::8], color='black', alpha=0.7, headwidth=4)
    axes[0].add_patch(Circle((x_c, y_c), D/2, color='black'))
    axes[0].set_title(f"Vitesse — Re = {Re_base:.0f} (convergé en {iter_conv} it.)")
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
    fig.colorbar(im1, ax=axes[0], label="||u|| (m/s)")

    im2 = axes[1].imshow(p, cmap='coolwarm', origin='lower', extent=[0, Lx, 0, Ly])
    axes[1].add_patch(Circle((x_c, y_c), D/2, color='black'))
    axes[1].set_title("Pression relative (Pa)")
    axes[1].set_xlabel("X (m)")
    fig.colorbar(im2, ax=axes[1], label="p (Pa)")
    
    plt.suptitle(f"Simulation de base — Cylindre D={D*100:.0f} cm, U₀={U0} m/s", fontsize=13)
    plt.tight_layout()
    plt.savefig("Cylindre_Base.png", dpi=150, bbox_inches='tight')
    plt.show()

def afficher_validation_cd(Re_list, Cd_exp, resultats_Cd):
    """Trace la courbe de comparaison du coefficient de traînée."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(Re_list, Cd_exp, 'o--', color='#1D9E75', lw=2, ms=8, label='Valeurs expérimentales')
    ax.semilogx(Re_list, resultats_Cd, 's-', color='#E24B4A', lw=2, ms=8, label='Simulation N.-S. 2D')
    
    for re, cd in zip(Re_list, resultats_Cd):
        ax.annotate(f'{cd:.2f}', (re, cd), xytext=(8, 5), textcoords='offset points', fontsize=10, color='#E24B4A')
        
    ax.set_xlabel('Nombre de Reynolds Re')
    ax.set_ylabel('Coefficient de traînée Cd')
    ax.set_title('Validation Cd — Cylindre circulaire D = 10 cm', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 2.0])
    plt.tight_layout()
    plt.savefig("Cylindre_Validation_Cd.png", dpi=150, bbox_inches='tight')
    plt.show()

def afficher_etude_parametrique(Re_etude, champs_Re, longueurs_recirculation, X, Y, x_c, y_c, D, Lx, Ly):
    """Affiche les résultats de l'évolution des tourbillons en fonction de Re."""
    # 1. Graphique des champs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, Re in enumerate(Re_etude):
        u_, v_, p_, U_in = champs_Re[Re]
        vitesse = np.sqrt(u_**2 + v_**2)
        
        im1 = axes[0, col].imshow(vitesse, cmap='jet', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=U_in*1.5)
        axes[0, col].quiver(X[::8, ::8], Y[::8, ::8], u_[::8, ::8], v_[::8, ::8], color='black', alpha=0.6, headwidth=4)
        axes[0, col].add_patch(Circle((x_c, y_c), D/2, color='black'))
        axes[0, col].set_title(f'Vitesse — Re = {Re:.0e}')
        
        im2 = axes[1, col].imshow(p_, cmap='coolwarm', origin='lower', extent=[0, Lx, 0, Ly])
        axes[1, col].add_patch(Circle((x_c, y_c), D/2, color='black'))
        axes[1, col].set_title(f'Pression — Re = {Re:.0e}')
        
    plt.suptitle("Étude paramétrique — Formation des tourbillons")
    plt.tight_layout()
    plt.savefig("Cylindre_Reynolds_Champs.png", dpi=150)
    plt.show()

    # 2. Graphique de la zone de recirculation
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(Re_etude, longueurs_recirculation, 's-', color='#534AB7', lw=2, ms=9)
    ax.set_xlabel('Nombre de Reynolds Re'); ax.set_ylabel('Longueur Lr/D')
    ax.set_title('Taille de la zone de recirculation')
    ax.grid(True, alpha=0.3)
    plt.savefig("Cylindre_Recirculation.png", dpi=150)
    plt.show()