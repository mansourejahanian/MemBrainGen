import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import imageio
import numpy as np
import os

def center_crop(x, current_size, desired_size):
    start = int((current_size - desired_size)/2)
    return x[:,:, start:(start + desired_size), start:(start + desired_size)]

def convert(img, target_type_min, target_type_max, target_type):
	imin = img.min()
	imax = img.max()

	a = (target_type_max - target_type_min) / (imax - imin)
	b = target_type_max - a * imax
	new_img = (a * img + b).astype(target_type)

	return new_img


def save_image(img, subject, category, repeat, roi, postfix="", save_dir=""):

	if subject == 5:
		category_dict = {
			655: ('bodies', 2),
			981: ('bodies', 2),
		}
	else:
		category_dict = {
			9: ('animals', 1),
			146: ('animals', 1),
			207: ('animals', 1),
			281: ('animals', 1),
			285: ('animals', 1),
			295: ('animals', 1),
			346: ('animals', 1),
			348: ('animals', 1),
			355: ('animals', 1),
			385: ('animals', 1),
			931: ('foods', 1),
			934: ('foods', 1),
			937: ('foods', 1),
			938: ('foods', 1),
			943: ('foods', 1),
			948: ('foods', 1),
			950: ('foods', 1),
			954: ('foods', 1),
			962: ('foods', 1),
			963: ('foods', 1),
			457: ('humans', 2),
			655: ('humans', 2),
			834: ('humans', 2),
			842: ('humans', 2),
			981: ('humans', 2),
			460: ('places', 1),
			582: ('places', 1),
			598: ('places', 1),
			706: ('places', 1),
			718: ('places', 1),
			762: ('places', 1),
			970: ('places', 1),
			975: ('places', 1),
			977: ('places', 1),
			978: ('places', 1)
		}

	plt.figure()
	plt.imshow(img, aspect='equal')
	plt.tight_layout()
	plt.axis('off')
	
	if save_dir != "":
		output_dir = f"./data/neurogen_outputs/{save_dir}/S{subject:02d}/{roi}/"
	else:
		output_dir = f"./data/neurogen_outputs/S{subject:02d}/{roi}/"
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if postfix == "":
		postfix_str = ""
	else:
		postfix_str = f"_{postfix}"
	
	plt.imsave(output_dir + category_dict[category][0]  + '_C%04d'%category + '_repeat%d'%repeat + f'{postfix_str}.png', img, format='png')

	return


def save_gif(img, subject, category, repeat, roi):

	fig = plt.figure()
	plt.tight_layout()
	plt.axis('off')
	ima = []
	for cur in img:
		im = plt.imshow(cur, animated=True, aspect='equal')
		ima.append( [im] )
	ani = ArtistAnimation(fig, ima, interval=30, blit=True)
    
	output_dir = output_dir = './img/S%0d'%subject + '/ROI%02d'%roi + '/'
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	img_gif = convert(np.asarray(img), 0, 255, np.uint8)
	imageio.mimwrite(output_dir + 'C%04d'%category + '_repeat%d'%repeat + '.png', img_gif, fps=32)
		
	return
	
