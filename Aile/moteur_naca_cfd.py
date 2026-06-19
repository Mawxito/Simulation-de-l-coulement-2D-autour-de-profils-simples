"""
PROGRAMME PRINCIPAL - ÉTUDE DU PROFIL NACA 0012 (Projet PMI_Z7)
"""
import numpy as np
import moteur_naca_cfd as cfd
import affichage_naca as aff

print(f"  SIMULATION 2D — PROFIL NACA 0012 (C = {cfd.C*100:.0f} cm, U₀ = {cfd.U_in} m/s)")

# Paramètres de simulation
ANGLES_REF = [0, 5, 10, 15]
CL_EXP     = [0.0, 0.5, 1.0, 1.3]
tol        = 1e-6

dx, dy, X, Y = cfd.creer_grille()
dt = 0.0002

resultats = []
champs = {}
x_sep_vals = []

# --- Boucle sur les 4 angles d'attaque ---
for angle in ANGLES_REF:
    print(f"\nSimulation à α = {angle}°...")
    masque, xf, yf = cfd.generer_masque(X, Y, angle)
    
    u = np.full((200, 250), cfd.U_in); v = np.zeros((200, 250))
    p = np.zeros((200, 250));          b = np.zeros((200, 250))
    u[masque], v[masque] = 0.0, 0.0
    
    erreur, it = 1.0, 0
    while erreur > tol and it < 6000:
        un = u.copy()
        u, v, p, b = cfd.pas_navier_stokes_naca(u, v, p, b, dx, dy, dt, masque, cfd.U_in)
        erreur = np.max(np.abs(u - un)) / cfd.U_in
        it += 1
        
    Cl, Cd = cfd.calculer_efforts(p, masque, dx, dy, cfd.U_in)
    print(f" -> Convergé ({it} it) | Cl = {Cl:.3f} | Cd = {Cd:.3f}")
    
    # Détection réelle du point de décollement sur l'extrados
    sep_detect = cfd.detecter_separation(u, masque, X, Y, x_c=1.0, y_c=1.0, C=cfd.C)
            
    resultats.append({'angle': angle, 'Cl': Cl, 'Cd': Cd})
    champs[angle] = (u.copy(), v.copy(), p.copy(), masque.copy())
    x_sep_vals.append(sep_detect)

# --- Affichages ---
print("\nCréation des graphiques...")
angles_sim = [r['angle'] for r in resultats]
Cl_sim = [r['Cl'] for r in resultats]
Cd_sim = [r['Cd'] for r in resultats]

aff.afficher_validation_cl_cd(angles_sim, Cl_sim, Cd_sim, ANGLES_REF, CL_EXP)
aff.afficher_champs_vitesse_pression(ANGLES_REF, champs, resultats, X, Y, 3.0, 2.0)
aff.afficher_separation(ANGLES_REF, champs, resultats, X, Y, 3.0, 2.0, x_sep_vals)
aff.afficher_cp(ANGLES_REF, champs, cfd, x_sep_vals, X, Y)

print("\nSimulation terminée avec succès.")