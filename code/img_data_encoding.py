
import numpy as np
import pandas as pd
from PIL import Image
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

non_flood_images = [
    6, 10, 13, 33, 40, 57, 59, 62, 67, 74, 87, 115, 150, 158, 161, 164,
    170, 172, 178, 185, 199, 206, 226, 264, 265, 269, 280, 282, 290,
    300, 323, 326, 328, 342, 354, 396, 487, 501, 504
]

start_time = time.time()
label_list = []
img_list = []
for i in range(0, 505):
    if i % 25 == 0:
        print(f'Loading image {i} at time {round(time.time() - start_time)}')
    is_in_non_flood_images = int(i in non_flood_images)
    for j in range(0, 1024):
        with Image.open(f'../img/img_{i}_{j}.jpg') as image:
            img_list.append(np.array(image.getdata(), np.uint8))
            label_list.append(is_in_non_flood_images)

with open('../data/all_images.csv', 'w') as file:
	for i, image in enumerate(img_list):
		if i % 100 == 0:
			print(f'Writing line {i}...')
		write_string = ''
		for integer in image:
			write_string += str(integer) + ','
		write_string += str(label_list[i]) + '\n'
		file.write(write_string)
		
print(f'Done. Took {round(time.time() - start_time)} seconds.')

#df = pd.concat(
#    [pd.DataFrame(img_list), pd.DataFrame(label_list, columns=['label'])],
#    axis=1
#)
#df.to_csv('../data/all_images.csv')
