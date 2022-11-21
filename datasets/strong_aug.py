import cv2
import numpy as np
import random
class contrast_gamma():
    def __init__(self, gamma = 1.0):
        self.gamma = gamma
    
    def __call__(self, img, lbl):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(img, table), lbl
        
class contrast_linear():
    def __init__(self, alpha = 3, beta = 0):
        self.a = alpha
        self.b = beta
    def __call__(self, img, lbl):
            
        '''
        out[pixel] = alpha * image[pixel] + beta
        alpha is for contrast, beta is for brightness
        '''
        output = np.zeros(img.shape, img.dtype)
        h, w, ch = img.shape
        for y in range(h):
            for x in range(w):
                for c in range(ch):
                    output[y,x,c] = np.clip(self.a*img[y,x,c] + self.b, 0, 255)

        return output, lbl

class brightness(contrast_linear):
    def __init__(self, alpha=1, beta=20):
        super().__init__(alpha, beta)
    def __call__(self, img, lbl):
        return super().__call__(img, lbl)

class brightness_channel(contrast_linear):
    def __init__(self, alpha=1, beta=20):
        super().__init__(alpha, beta)
    
    def __call__(self, img, lbl):
            
        '''
        out[pixel] = alpha * image[pixel] + beta*channel/2
        alpha is for contrast, beta is for brightness
        '''
        output = np.zeros(img.shape, img.dtype)
        h, w, ch = img.shape
        for y in range(h):
            for x in range(w):
                for c in range(ch):
                    output[y,x,c] = np.clip(self.a*img[y,x,c] + self.b*(c+1)/2 , 0, 255)

        return output, lbl

class equalize():
    '''
    Not close to the paper's code at all (iaa.HistogramEqualization()) but closest compare to
    >> gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    >> equalized = cv2.equalizeHist(gray)
    >> clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    >> equalized = clahe.apply(gray)

    '''
    def __init__(self) -> None:
        pass
    def __call__(self, img, lbl):
        # 1. calclate hist
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # 2. normalize hist
        h, w = img.shape[:2]
        hist = hist/(h*w)

        # 3. calculate CDF
        cdf = np.cumsum(hist)
        s_k = (255 * cdf - 0.5).astype("uint8")
        equalized_img = cv2.LUT(img, s_k)
        return equalized_img, lbl

class hsv():            # need fix
    '''
    HSV transform in cv2 
    Very different from iaa.AddToHueAndSaturation(1, per_channel=True).augment_image(img)
    '''
    def __init__(self) -> None:
        pass
    def __call__(self, img, lbl) :
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img,lbl

class invert_channel():
    def __init__(self) -> None:
        pass
    def __call__(self, image, mask):
        assert image.shape[2] == 3
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        return image, mask

class blur():               #need to fix parameters
    '''
    Kernel must be odd number
    Border Type can be [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv.BORDER_REFLECT, cv2.BORDER_WRAP,
    cv2.BORDER_REFLECT_101, cv2.BORDER_TRANSPARENT, cv2.BORDER_REFLECT101, cv2.BORDER_DEFAULT, cv2.BORDER_ISOLATED]
    '''
    def __init__(self, kernel_size = (7,7), borderType = cv2.BORDER_DEFAULT ) :
        self.kernel = kernel_size
        self.border = borderType
    def __call__(self, img, lbl):
        gau = cv2.GaussianBlur(img,self.kernel,self.border)
        return gau, lbl

class noise_gau():
    '''
    gaussian_img = image * 0.75 + 0.25 * gaussian + 0.25
    gaussian = np.random.normal(loc=mean, scale = std, size = (shape[0], shape[1], 1)).astype(np.float32)

    '''
    def __init__(self, mean = 0, std = 7, gamma = 0.25, alpha = 0.75 ):
        self.mean = mean
        self.std = std
        self.gamma = gamma
        self.alpha = alpha
    def __call__(self, image, lbl):
        image = image.astype(np.float32)
        shape = image.shape[:2]

        mean = self.mean
        sigma = self.std
        gamma = self.gamma
        alpha = self.alpha
        beta = 1 - alpha

        gaussian = np.random.normal(loc=mean, scale = 7, size = (shape[0], shape[1], 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        #gaussian_img = image * 0.75 + 0.25 * gaussian + 0.25
        gaussian_img = cv2.addWeighted(image, alpha, beta * gaussian, beta, gamma)
        gaussian_img = np.uint8(gaussian_img)

        return gaussian_img, lbl

class noise_pos():
    '''
    The same with gauss noise, but replace by poisson
    '''
    def __init__(self, lam = 100, gamma = 0.25, alpha = 0.75):
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
    def __call__(self, image, lbl):
        image = image.astype(np.float32)
        shape = image.shape[:2]

        lam = self.lam
        gamma = self.gamma
        alpha = self.alpha
        beta = 1 - alpha

        poisson = np.random.poisson(lam=lam, size = (shape[0], shape[1], 1)).astype(np.float32)
        poisson = np.concatenate((poisson, poisson, poisson), axis = 2)
        
        poisson_img = cv2.addWeighted(image, alpha, beta * poisson, beta, gamma)
        poisson_img = np.uint8(poisson_img)

        return poisson_img, lbl

class Channel_shuffle():
    def __init__(self) -> None:
        pass
    def __call__(self, img, lbl):
        lst = [0,1,2]
        random.shuffle(lst)
        return np.stack((img[:,:,lst[0]], img[:,:,lst[1]], img[:,:,lst[2]]), axis=-1), lbl

class Dropout():
    def __init__(self,rate = 0.05):
        self.rate = rate
    def __call__(self, img, lbl):
        size = img.shape[0] * img.shape[1] * img.shape[2]
        nums = np.ones(size)
        nums[:int(size*self.rate)] = 0
        np.random.shuffle(nums)
        nums = nums.reshape(img.shape)

        img_drop = np.multiply(img,nums).astype(int)
        lbl_drop = np.where(nums == 1,lbl, 255)
        return img_drop, lbl_drop

class Coarse_dropout():
    def __init__(self, box = (3,4), rate = 0.1):
        self.box = box
        self.rate = rate/ (box[0] * box[1])
    def __call__(self, img, lbl):
        size = img.shape[0] * img.shape[1]
        nums = np.ones(size)
        nums[:int(size*self.rate)] = 0
        np.random.shuffle(nums)
        nums = nums.reshape((img.shape[0], img.shape[1]))
        a,b = np.where(nums == 0)
        for i in range(a.shape[0]):
            img[a[i]: min(a[i]+ self.box[0] , img.shape[0]), b[i]: min(b[i] + self.box[1], img.shape[1]), :] = 0
            lbl[a[i]: min(a[i]+ self.box[0] , img.shape[0]), b[i]: min(b[i] + self.box[1], img.shape[1]), :] = 255
        return img, lbl
class Multiply():
    def __init__(self, low = 0, high = 1):
        self.low = low
        self.high = high
    def __call__(self, img, lbl):
        random = np.random.uniform(self.low,self.high ,size = img.shape)

        img_mul = np.multiply(img,random).astype(int)
        return img_mul, lbl

class salt_pepper():
    def __init__(self, min= 300, max= 10000):
        #drop a number of pixels between min and max with salt_pepper
        self.min = min
        self.max = max
    def __call__(self, img, lbl):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Getting the dimensions of the image
        row , col = img.shape
        
        # Randomly pick some pixels in the
        # image for coloring them white
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(self.min, self.max)
        for i in range(number_of_pixels):
        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
            
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
            
            # Color that pixel to white
            img[y_coord][x_coord] = 255
            
        # Randomly pick some pixels in
        # the image for coloring them black
        # Pick a random number between 300 and 10000
        number_of_pixels = random.randint(self.min , self.max)
        for i in range(number_of_pixels):
        
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)
            
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)
            
            # Color that pixel to black
            img[y_coord][x_coord] = 0
            lbl[y_coord, x_coord, :] = 255 
        return img,lbl

class solarize():
    def __init__(self, threshold = 128):
        self.thresh = threshold
    def __call__(self, img, lbl) :
        """Invert all pixel values above a threshold.
        Args:
            img (numpy.ndarray): The image to solarize.
            threshold (int): All pixels above this greyscale level are inverted.
        Returns:
            numpy.ndarray: Solarized image.
        """
        max_val = 255
        
        lut = [(i if i < self.thresh else max_val - i) for i in range(max_val + 1)]
        prev_shape = img.shape
        img = cv2.LUT(img, np.array(lut))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img, lbl


