import cv2
import numpy as np
from collections import Counter

class ReconstructPuzzle:
    """
    Takes x-y coordinates as the alphabets name and use them in loading
    the alphabets images from disk.
    """

    def __init__(self, coordinates):
        self.coordinates = coordinates
    
    def load_alphabets_array(self, puzzle_alphabets_dir):
        """
        Loads all alphabets images.
        params: 
            puzzle_alphabets_dir - the directory containing all the extracted alphabets.
        return: 
            a 3D-array (N x H x W) of all the images.
        """
        puzzle_img_array = []
        for coord in self.coordinates:
            coord = str(coord)
            c1, c2 = coord.split(",")

            coord = c1.strip()+","+c2.strip()

            path = puzzle_alphabets_dir+f"/{coord}.jpg"

            img = cv2.imread(path, 0)
            puzzle_img_array.append(cv2.resize(img, (20,20), interpolation=cv2.INTER_AREA))

        puzzle_img_array = np.stack(puzzle_img_array)

        return puzzle_img_array


    def reconstruct_puzzle(self, puzzle_alphabets_dir, offset=10):
        """
        Takes all the stacked alphabets array and builds into a single a single 2D-Image.
        params: 
            puzzle_alphabets_dir: the directory containing all the extracted alphabets.
            offset: the distance between each alphabets array in both x-y direction in the
                    constructed image.
        return: 
            A 2D-Image containing all the original puzzle alphabets in the right order.
        """

        images = self.load_alphabets_array(puzzle_alphabets_dir)

        y_points = [point[1] for point in self.coordinates]
        n_cols = max(Counter(y_points).values())

        n_images, image_height, image_width = images.shape
        n_rows = n_images // n_cols

        puzzle_dim = (image_height*n_rows, image_width*n_cols)
        
        dest_height, dest_width = puzzle_dim
        
        dest_height += offset * (n_cols+1)
        dest_width += offset * (n_rows+1)
        
        puzzle_img = np.ones((dest_height, dest_width))
        puzzle_img[:, :] = 0
        
        c_1 = 0 
        c_2 = 0 
        
        start_width = c_1 + offset
        end_width = start_width + image_width  
        
        start_height = c_2 + offset
        end_height = start_height + image_width
        
        cols_count = 1
        for img in images:

            if cols_count < n_cols:
                puzzle_img[start_height:end_height, \
                        start_width:end_width] = img
                c_1 = end_width 
                start_width = c_1 + offset
                end_width = start_width + image_width   
                
            elif cols_count == n_cols:
                
                puzzle_img[start_height:end_height, \
                        start_width:end_width] = img
                start_height = end_height + offset
                end_height = start_height + image_width
                
                start_width = offset
                end_width = start_width + image_width
                
                cols_count = 0

            cols_count += 1
        return puzzle_img.astype(np.uint8)
