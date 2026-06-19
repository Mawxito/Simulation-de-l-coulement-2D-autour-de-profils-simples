"""
PROGRAMME PRINCIPAL - ÉTUDE DU CYLINDRE (Projet PMI_Z7)
Ce script orchestre les simulations en faisant appel au moteur CFD 
et au module d'affichage externe.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse
from scipy.signal import find_peaks

# Importation des modules personnalisés
import moteur_cfd as cfd
import affichage_graphiques as aff

# ==============================================================================
# PARAMÈTRES GLOBAUX DU PROJET
# ==============================================================================
D = 0.1          # Diamètre du cylindre (m)
U0 = 10.0        # Vitesse d'entrée imposée par la consigne (m/s)
Re_base = U0 * D / cfd.nu             # Calcul du Reynolds de référence (≈ 66 667)

print("  DÉMARRAGE DU PROGRAMME PRINCIPAL - ÉTUDE CFD")
print(f"  Re de référence = {Re_base:.0f}")

# ==============================================================================
# SECTION 1 — SIMULATION DE BASE (Régime stationnaire)
# ==============================================================================
print("\n[Section 1] Lancement de la simulation de base (Convergence < 1e-6)...")

nx, ny = 200, 200
Lx, Ly = 2.0, 2.0
dx, dy, X, Y = cfd.creer_grille(nx, ny, Lx, Ly)
x_c, y_c = 0.5, 1.0

masque_c = cfd.masque_cylindre(X, Y, x_c, y_c, D)
dt = 0.5 * min(dx, dy) / U0   # CFL ≈ 0.5

u = np.full((ny, nx), U0); v = np.zeros((ny, nx))
p = np.zeros((ny, nx));    b = np.zeros((ny, nx))
cfd.appliquer_limites(u, v, U0, masque_c)

tol = 1e-6
iter_conv = 0

# Boucle de convergence
for it in range(8000):
    u_old = u.copy()
    u, v, p, b = cfd.pas_navier_stokes(u, v, p, b, dx, dy, dt, cfd.nu, cfd.rho, masque_c, U0)
    erreur = np.max(np.abs(u - u_old)) / U0
    iter_conv += 1
    
    if erreur < tol:
        print(f"  -> Convergé avec succès en {iter_conv} itérations !")
        break

u_base, v_base, p_base = u.copy(), v.copy(), p.copy()

# Appel au module d'affichage
aff.afficher_simulation_base(u_base, v_base, p_base, X, Y, x_c, y_c, D, Lx, Ly, U0, Re_base, iter_conv)


# ==============================================================================
# SECTION 2 — VALIDATION DU COEFFICIENT DE TRAÎNÉE (Cd)
# ==============================================================================
print("\n[Section 2] Validation du Cd avec la littérature...")

Re_list   = [1e3, 1e4, 1e5]
Cd_exp    = [1.0, 1.1, 1.2]   # Données expérimentales
resultats_Cd = []

for Re in Re_list:
    U_in = Re * cfd.nu / D
    dt_  = 0.5 * dx / U_in
    nt_  = int(4 * (Lx / U_in) / dt_) # Temps pour 4 traversées
    
    u_ = np.full((ny, nx), U_in); v_ = np.zeros((ny, nx))
    p_ = np.zeros((ny, nx));      b_ = np.zeros((ny, nx))
    cfd.appliquer_limites(u_, v_, U_in, masque_c)
    
    Cd_hist = []
    for n in range(nt_):
        u_, v_, p_, b_ = cfd.pas_navier_stokes(u_, v_, p_, b_, dx, dy, dt_, cfd.nu, cfd.rho, masque_c, U_in)
        if n > int(0.8 * nt_) and n % 10 == 0:
            Cd_hist.append(cfd.calculer_Cd(p_, masque_c, dx, dy, cfd.rho, U_in, D))
            
    Cd_sim = np.mean(Cd_hist)
    resultats_Cd.append(Cd_sim)
    print(f"  Re = {Re:.0e} | Cd simulé = {Cd_sim:.3f} | Erreur = {abs(Cd_sim - Cd_exp[Re_list.index(Re)]) / Cd_exp[Re_list.index(Re)] * 100:.1f}%")

aff.afficher_validation_cd(Re_list, Cd_exp, resultats_Cd)


# ==============================================================================
# SECTION 3 — ÉTUDE PARAMÉTRIQUE (Évolution des tourbillons)
# ==============================================================================
print("\n[Section 3] Étude de la taille des tourbillons de sillage...")

Re_etude = [1e3, 1e4, 1e5]
champs_Re = {}
longueurs_recirculation = []

for Re in Re_etude:
    U_in = Re * cfd.nu / D
    dt_  = 0.5 * dx / U_in
    nt_  = int(3 * (Lx / U_in) / dt_)
    
    u_ = np.full((ny, nx), U_in); v_ = np.zeros((ny, nx))
    p_ = np.zeros((ny, nx));      b_ = np.zeros((ny, nx))
    cfd.appliquer_limites(u_, v_, U_in, masque_c)
    
    for _ in range(nt_):
        u_, v_, p_, b_ = cfd.pas_navier_stokes(u_, v_, p_, b_, dx, dy, dt_, cfd.nu, cfd.rho, masque_c, U_in)
        
    champs_Re[Re] = (u_.copy(), v_.copy(), p_.copy(), U_in)
    Lr = cfd.calculer_longueur_recirculation(u_, X, Y, x_c, D, y_c)
    longueurs_recirculation.append(Lr)
    print(f"  Re = {Re:.0e} -> Lr/D = {Lr:.2f}")

aff.afficher_etude_parametrique(Re_etude, champs_Re, longueurs_recirculation, X, Y, x_c, y_c, D, Lx, Ly)


# ==============================================================================
# SECTION 4 — ANALYSE VIV (Allée de Von Kármán & Strouhal)
# ==============================================================================
print("\n[Section 4] Analyse du détachement tourbillonnaire (VIV)...")

Re_viv  = 1e4
U_viv   = Re_viv * cfd.nu / D
Lx_v, Ly_v = 4.0, 2.0
dx_v, dy_v, X_v, Y_v = cfd.creer_grille(200, 200, Lx_v, Ly_v)
x_cv, y_cv = 1.0, 1.0
masq_v = cfd.masque_cylindre(X_v, Y_v, x_cv, y_cv, D)

dt_v  = 0.4 * dx_v / U_viv
nt_v  = int(20 * (Lx_v / U_viv) / dt_v)

u_v = np.full((200, 200), U_viv); v_v = np.zeros((200, 200))
p_v = np.zeros((200, 200));       b_v = np.zeros((200, 200))
cfd.appliquer_limites(u_v, v_v, U_viv, masq_v)

# Perturbation initiale pour forcer l'asymétrie
v_v[102, :] += 0.01 * U_viv   

sonde_x, sonde_y = int((x_cv + 3*D) / dx_v), int((y_cv + D) / dy_v)
hist_t, hist_v = [], []

print("  Calcul en cours...")
for n in range(nt_v):
    u_v, v_v, p_v, b_v = cfd.pas_navier_stokes(u_v, v_v, p_v, b_v, dx_v, dy_v, dt_v, cfd.nu, cfd.rho, masq_v, U_viv)
    if n > int(0.3 * nt_v):
        hist_t.append(n * dt_v)
        hist_v.append(float(v_v[sonde_y, sonde_x]))

# --- Analyse FFT et Strouhal ---
t_arr, v_arr = np.array(hist_t), np.array(hist_v)
freq  = np.fft.rfftfreq(len(v_arr), d=(t_arr[1]-t_arr[0]))
ampli = np.abs(np.fft.rfft(v_arr - np.mean(v_arr)))
f_vortex = freq[np.argmax(ampli[1:])+1]
St_sim = f_vortex * D / U_viv
print(f"  Fréquence détectée = {f_vortex:.3f} Hz | Nombre de Strouhal = {St_sim:.3f}")


# ==============================================================================
# SECTION 5 — ATTÉNUATION (Cylindre Elliptique)
# ==============================================================================
print("\n[Section 5] Test d'atténuation avec un cylindre elliptique...")

a_ell, b_ell = D, D / 2
masq_ell = cfd.masque_ellipse(X_v, Y_v, x_cv, y_cv, a=a_ell, b=b_ell)

u_e = np.full((200, 200), U_viv); v_e = np.zeros((200, 200))
p_e = np.zeros((200, 200));       b_e = np.zeros((200, 200))
cfd.appliquer_limites(u_e, v_e, U_viv, masq_ell)
v_e[102, :] += 0.01 * U_viv

hist_v_e = []
for n in range(nt_v):
    u_e, v_e, p_e, b_e = cfd.pas_navier_stokes(u_e, v_e, p_e, b_e, dx_v, dy_v, dt_v, cfd.nu, cfd.rho, masq_ell, U_viv)
    if n > int(0.3 * nt_v):
        hist_v_e.append(float(v_e[sonde_y, sonde_x]))

v_e_arr = np.array(hist_v_e)
rms_cyl = np.sqrt(np.mean(v_arr**2))
rms_ell = np.sqrt(np.mean(v_e_arr**2))
reduction = (1 - rms_ell / rms_cyl) * 100
print(f"  Réduction des oscillations transverses : {reduction:.1f}%")


# ==============================================================================
# AFFICHAGE FINAL (Analyse VIV & Atténuation)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_arr, v_arr, color='#E24B4A', lw=1.2, label=f'Circulaire (RMS = {rms_cyl:.4f})')
axes[0].plot(t_arr, v_e_arr, color='#1D9E75', lw=1.2, label=f'Elliptique (RMS = {rms_ell:.4f})')
axes[0].set_xlabel('Temps (s)'); axes[0].set_ylabel('Vitesse transverse à la sonde (m/s)')
axes[0].set_title(f'Atténuation VIV — Réduction de {reduction:.1f}%')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

vitesse_e = np.sqrt(u_e**2 + v_e**2)
im = axes[1].imshow(vitesse_e, cmap='jet', origin='lower', extent=[0, Lx_v, 0, Ly_v], vmin=0, vmax=U_viv*1.5)
axes[1].add_patch(Ellipse((x_cv, y_cv), 2*a_ell, 2*b_ell, color='black'))
axes[1].set_title(f'Sillage de l\'ellipse (a/b = {a_ell/b_ell:.1f})')
axes[1].set_xlabel('X (m)')
fig.colorbar(im, ax=axes[1], label='||u|| (m/s)')

plt.tight_layout()
plt.savefig("Cylindre_Analyse_Finale.png", dpi=150)
plt.show()

print("\n=== PROGRAMME TERMINÉ AVEC SUCCÈS ===")