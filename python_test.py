import numpy as np
import matplotlib


path = r'D:\ImageNet\clickme_val\0.npz'
data = np.load(path)
data_file =data.files
print(data['image'])
