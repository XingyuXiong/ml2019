from PIL import Image
import matplotlib.pyplot as plt
import sys,os
import numpy as np

work_path=sys.path[0]
pil_im=Image.open(work_path+r'/yaleB01/yaleB01_P00A+000E+00.pgm')


fig=plt.figure('person')
ax=fig.add_subplot(221)
ax.imshow(pil_im)
plt.show()
print(np.array(pil_im))