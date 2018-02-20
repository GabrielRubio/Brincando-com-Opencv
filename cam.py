import cv2
import numpy as np
## We need to import matplotlib to create our histogram plots
#from matplotlib import pyplot as plt

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, image = cam.read()
        if mirror: 
            image = cv2.flip(image, 1)
            
            # ######################################### desenho de objetos #########################################
            cv2.rectangle(image, (100,100), (300,250), (127,50,127), 5)
            # cv2.circle(image, (350, 350), 100, (15,75,50), 2) 
            
            # ######################################### Imagem original #########################################
            cv2.imshow('Original image', image)
            
            # ######################################### Manipulacao de cor #########################################
            # ## Mudanca no padrao de cor RGB -> escala de cinza  
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Gray image', gray_image)

            # ## Mudanca no padrao de cor de RGB -> HSV 
            # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # cv2.imshow('HSV image', hsv_image)
            # cv2.imshow('Hue channel', hsv_image[:, :, 0])
            # cv2.imshow('Saturation channel', hsv_image[:, :, 1])
            # cv2.imshow('Value channel', hsv_image[:, :, 2])
            
            # # Alteracao na escala de cor em RGB
            # B, G, R = cv2.split(image)
            # # Let's amplify the blue color
            # merged = cv2.merge([B, G-100, R+100])
            # cv2.imshow("Merged with Blue Amplified", merged) 
    
            # zeros = np.zeros(image.shape[:2], dtype = "uint8")
            # cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
            # cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
            # cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
            
            # ######################################### Translacao da imagem #########################################
            # height, width = image.shape[:2] # pegando medidas da imagem
            # quarter_height, quarter_width = height/4, width/4  # definindo para onde vai transladar
            # #       | 1 0 Tx |
            # #  T  = | 0 1 Ty |
            # T = np.float32([[1, 0, quarter_width], [0, 1,quarter_height]])  # T eh a matriz de translacao
            # img_translation = cv2.warpAffine(image, T, (width, height)) # metodo para translacao
            # cv2.imshow('Translation', img_translation)

            # ########################################## Rotacao de imagem #########################################
            # height, width = image.shape[:2] # pegando medidas da imagem
            # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 75, .5) # matriz de rotacao  da imagem
            # rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height)) # metodo usando a matriz de rotacao
            # cv2.imshow('Rotated Image', rotated_image)
            
            # rotated_image = cv2.transpose(image) # transpoe a imagem
            # cv2.imshow('Rotated Image - Method 2', rotated_image)
            
            # flipped = cv2.flip(image, 1) #espelha a imagem
            # cv2.imshow('Horizontal Flip', flipped) 

            # ########################################## Interpolacao e escala #########################################
            # image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
            # cv2.imshow('Scaling - Linear Interpolation', image_scaled)

            # # Let's double the size of our image
            # img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
            # cv2.imshow('Scaling - Cubic Interpolation', img_scaled)

            # # Let's skew the re-sizing by setting exact dimensions
            # img_scaled = cv2.resize(image, (400, 900), interpolation = cv2.INTER_AREA)
            # cv2.imshow('Scaling - Skewed Size', img_scaled) 

            # # outra maneira de escalar
            # smaller = cv2.pyrDown(image)
            # larger = cv2.pyrUp(smaller)
            # cv2.imshow('Smaller ', smaller )
            # cv2.imshow('Larger ', larger )

            # ########################################## Cortando a imagem #########################################
            # height, width = image.shape[:2]
            # start_row, start_col = int(height * .0), int(width * .0) # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
            # end_row, end_col = int(height * .50), int(width * .50)     # Let's get the ending pixel coordinates (bottom right)
            # cropped = image[start_row:end_row , start_col:end_col]     # Simply use indexing to crop out the rectangle we desire
            # cv2.imshow("Cropped Image 1", cropped) 

            # start_row, start_col = int(height * .50), int(width * .0) # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
            # end_row, end_col = int(height * 1.0), int(width * .50)     # Let's get the ending pixel coordinates (bottom right)
            # cropped = image[start_row:end_row , start_col:end_col]     # Simply use indexing to crop out the rectangle we desire
            # cv2.imshow("Cropped Image 2 ", cropped)

            # start_row, start_col = int(height * .0), int(width * .50) # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
            # end_row, end_col = int(height * .50), int(width * 1.0)     # Let's get the ending pixel coordinates (bottom right)
            # cropped = image[start_row:end_row , start_col:end_col]     # Simply use indexing to crop out the rectangle we desire
            # cv2.imshow("Cropped Image 3 ", cropped)

            # start_row, start_col = int(height * .50), int(width * .50) # Let's get the starting pixel coordiantes (top  left of cropping rectangle)
            # end_row, end_col = int(height * 1.0), int(width * 1.0)     # Let's get the ending pixel coordinates (bottom right)
            # cropped = image[start_row:end_row , start_col:end_col]     # Simply use indexing to crop out the rectangle we desire
            # cv2.imshow("Cropped Image 4 ", cropped)

            # # ########################################## Convolucao e filtros #########################################
            # kernel_3x3 = np.ones((3, 3), np.float32) / 9 # Creating our 3 x 3 kernel
            # blurred = cv2.filter2D(image, -1, kernel_3x3)# We use the cv2.fitler2D to conovlve the kernal with an image 
            # cv2.imshow('3x3 Kernel Blurring', blurred)
            
            # vertical = np.array([[1.,0,-1.],[1.,0,-1.],[1.,0,-1.]])
            # horizontal = np.array([[1.,1.,1.],[0,0,0],[-1.,-1.,-1.]])
            
            # blurred_v = cv2.filter2D(image, -1, vertical)# We use the cv2.fitler2D to conovlve the kernal with an image 
            # cv2.imshow('Sobel vertical', blurred_v)
            
            # blurred_h = cv2.filter2D(image, -1, horizontal)
            # cv2.imshow('Sobel horizontal', blurred_h)

            # border = cv2.add(blurred_h,blurred_v)
            # cv2.imshow('Sobel border', border)

            # kernel_7x7 = np.ones((7, 7), np.float32) / 49 # Creating our 7 x 7 kernel
            # blurred2 = cv2.filter2D(image, -1, kernel_7x7)
            # cv2.imshow('7x7 Kernel Blurring', blurred2)

            # blur = cv2.blur(image, (3,3)) # Borramento pela media de um quadro (3,3)
            # cv2.imshow('Averaging', blur)

            # Gaussian = cv2.GaussianBlur(image, (7,7), 0) # Instead of box filter, gaussian kernel
            # cv2.imshow('Gaussian Blurring', Gaussian)

            # median = cv2.medianBlur(image, 5)  # element is replaced with this median value
            # cv2.imshow('Median Blurring', median)

            # bilateral = cv2.bilateralFilter(image, 9, 75, 75) # Bilateral is very effective in noise removal while keeping edges sharp
            # cv2.imshow('Bilateral Blurring', bilateral)

            # # ########################################## Bordas #########################################
            # kernel_sharpening = np.array([[-1,-1,-1], 
            #                               [-1,9,-1], 
            #                               [-1,-1,-1]])

            # # applying different kernels to the input image
            # sharpened = cv2.filter2D(image, -1, kernel_sharpening)
            # cv2.imshow('Image Sharpening', sharpened)

            # sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            # sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            # cv2.imshow('Sobel X', sobel_x)
            # cv2.imshow('Sobel Y', sobel_y)

            # sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
            # cv2.imshow('sobel_OR', sobel_OR)

            # laplacian = cv2.Laplacian(image, cv2.CV_64F)
            # cv2.imshow('Laplacian', laplacian)

            # canny = cv2.Canny(image, 100, 150)
            # cv2.imshow('Canny', canny)

            if cv2.waitKey(1) == 27: 
                break  # esc to quit
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()

