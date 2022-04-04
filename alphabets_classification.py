import cv2
import pickle
import numpy as np

class AlphabetsClassifier:
    """
    Takes a puzzle image, extract all the alphabets present in the puzzle using 
    OpenCV connectedComponent module and uses a SVC (a Machine Learning Model) to classify
    the arrays of alphabets extracted.

    params:
        model_src: the classify model location.
        puzzle_src: location of the puzzle image.
    return:
        Predictions and the x-y coordinates of the alphabets in the puzzle. 
    
    """
    
    def __init__(self, model_src, puzzle_src):
    
        self.puzzle_src = puzzle_src

        self.model = pickle.load(open(model_src, "rb"), encoding="utf-8")
        self.pca = pickle.load(open("PCA.pkl", "rb"), encoding="utf-8")
        
    def __image_connected_components_statistics(self):
        """
        Computes image connected component statistics.
        """
        puzzle_image = cv2.imread(self.puzzle_src) 
        gray = cv2.cvtColor(puzzle_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        image_stats = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
        (num_labels, _ , stats, _) = image_stats
        
        return stats, num_labels, puzzle_image
    
    

    def alpha_image_classification(self):
        """
        Takes in the information from the computed statistics to determine the
        region most likely contatining an alphabets and pass those region to a 
        Support vector classify.
        """
        alphabets_array = []
        coordinates = []

        stats, num_labels, puzzle_image = self.__image_connected_components_statistics()
        for i in range(0, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            keepWidth = w >= 1

            img_height = puzzle_image.shape[0]
            
            if (i > 0 and keepWidth and \
            h < abs(img_height-70) and area > 5):

                alpha_img = puzzle_image[y:y+h, x:x+w]

                top=bottom= int((alpha_img.shape[0]-1) / 2 * 1.5)
                left=right= int((alpha_img.shape[1]-1))
                if w < 5:
                    left=right=top
                alpha_img = cv2.copyMakeBorder(alpha_img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

                alpha_img = cv2.cvtColor(alpha_img, cv2.COLOR_BGR2GRAY)
                alpha_img = cv2.resize(alpha_img, (38,38), interpolation=cv2.INTER_AREA)
                alpha_img = alpha_img.reshape(38*38)

                
                alphabets_array.append(alpha_img)
                coordinates.append((x,y))

        alphabets_array = self.pca.transform(np.stack(alphabets_array))
        
        predictions = self.model.predict(alphabets_array)
        
        return predictions, coordinates
                
                

predictions, coordinates = AlphabetsClassifier("SVMclassify.pkl", "test_images/t_2.jpeg").alpha_image_classification()


        


    