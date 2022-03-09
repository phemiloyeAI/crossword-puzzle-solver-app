class PuzzleAlphabetsExtract:
    
    def __init__(self, src, dir_name):
        self.src = src
        self.dir_name = dir_name
    
    def __image_connected_components_statistics(self):
        puzzle_image = cv2.imread(self.src) 
        gray = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        image_stats = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
        (num_labels, _ , stats, _) = image_stats
        
        return stats, num_labels, puzzle_image
    
    def extract_alphabets_arrays(self):
        stats, num_labels, puzzle_image = self.__image_connected_components_statistics()
        counts = 0
        for i in range(0, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            keepWidth = w>2 or w == 2
            img_height = puzzle_image.shape[0]
            
            if (i > 0 and keepWidth and \
            h < abs(img_height-200) and area > 10):

                char = puzzle_image[y:y+h, x:x+w]

                cv2.imwrite(f"{self.dir_name}/({x},{y}).jpg", char)
                counts +=1

        print(f"{counts} alphabets saved to {self.dir_name}!")
