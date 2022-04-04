import cv2

class PuzzleAlphabetsExtract:
    """
    This class takes in a puzzle image and extracts all the alphabets present in it
    using OpenCV ConnectedComponent function.

    params: 
        src: puzzle image source.
        dest_dir: the directory where all alphabets extracted are saved.
    return: 
        None
    """
    
    def __init__(self, src, dest_dir=None):
        self.src = src
        self.dest_dir = dest_dir
    
    def __image_connected_components_statistics(self):
        """
        Computes the statistic to be used in extracting alphabets.

        return: 
            image statistics, image
        """
        puzzle_image = cv2.imread(self.src) 
        gray = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        image_stats = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
        (num_labels, _ , stats, _) = image_stats
        
        return stats, num_labels, puzzle_image
    
    def extract_alphabets_arrays(self):
        """
        Extracts all regions identified as components(alphabets) and save them to disk.
        """
        
        stats, num_labels, puzzle_image = self.__image_connected_components_statistics()
        counts = 0
        c=0
        for i in range(0, num_labels):
            
            #extract the individual parameters 
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            keepWidth = w >= 1

            img_height = puzzle_image.shape[0]
            
            # this conditional statement ensures only alphabets array are indexed and saved
            if (i > 0 and keepWidth and \
            h < abs(img_height-70) and area > 5):

                char = puzzle_image[y:y+h, x:x+w]

                top=bottom= int((char.shape[0]-1) / 2 * 1.5)
                left=right= int((char.shape[1]-1))

                
                if w < 5:
                    left=right=top
                    c+=1

                #give padding to the cropped alphabets
                char = cv2.copyMakeBorder(char, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                #save the alphabets to the given destination directory
                cv2.imwrite(f"{self.dest_dir}/({x},{y}).jpg", char)

                counts +=1

        print(f"{counts} alphabets saved to {self.dest_dir}!")




#extract all alphabets in this given puzzle image
puzz_image = PuzzleAlphabetsExtract("test_images/t_1.jpeg", "test_images/t1_puzz_chars").extract_alphabets_arrays()


