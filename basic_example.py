import logging

import numpy as np
from matplotlib import pyplot as plt

from roipoly import RoiPoly

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Chargement de l'image
img = plt.imread('./data/MRI.jpg')

# Affichage de l'image
fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.title("Clic gauche : segments de droite (tracé de la ROI)" + "\n" + "Clic droit : fermeture de la ROI")
plt.show(block=False)

# Tracé de la première ROI
roi1 = RoiPoly(color='r', fig=fig)

# Affichage de l'image avec la première ROI
fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.colorbar()
roi1.display_roi()
plt.title('draw second ROI')
plt.show(block=False)

# Tracé de la deuxième ROI
roi2 = RoiPoly(color='b', fig=fig)

# Affichage de l'image avec les deux ROI et leurs valeurs moyennes
plt.imshow(img, cmap='gray')
plt.colorbar()
for roi in [roi1, roi2]:
    roi.display_roi()
    roi.display_mean(img)
plt.title('The two ROIs')
plt.show()

# Sauvegarde des masques (arrays numpy) sous forme de fichiers .npy 
np.save('./mask1.npy', roi1.get_mask(img))
np.save('./mask2.npy', roi2.get_mask(img))
np.save('./mask.npy', roi1.get_mask(img) + roi2.get_mask(img))

# Affichage des deux masques correspondants
plt.imshow(roi1.get_mask(img) + roi2.get_mask(img), cmap="gray")
plt.title('ROI masks of the two ROIs')
plt.show()