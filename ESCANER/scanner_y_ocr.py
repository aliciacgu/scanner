import cv2
import numpy as np
import pytesseract

#indicar donde esta almacenado tesseract
pytesseract.pytesseract.tesseract_cmd = r'tesseractfolder\tesseract'

#funcion para orderar los vertices y asi poder hacer la transformacion de perspectiva sin cuidar el orden
def ordenar_puntos(puntos):
    #quitar el doble corchete que trae la lista de coordenadas encontradas, agregandolos a otra lista
    n_puntos = np.concatenate([puntos[0], puntos[1],puntos[2],puntos[3]]).tolist()
    
    #ordena los puntos en base a 'y' de cada coordenada de mayor a menor (arriba a abajo)
    #key es la especificacion de como queremos que se ordene
        #lambda es una funcion anonima, donde indicamos que se seleccione la segunda posicion de la coordenada, es decir, la 'y'
    #Al ordenarlos quedarán los puntos de arriba primeros y los puntos de abajo al final
    y_order=sorted(n_puntos,key=lambda n_puntos:n_puntos[1])
    
    #ordenamos en 'x' de izquierda a derecha DE LOS DOS PUNTOS DE ARRIBA
    x1_order= y_order[:2]
    x1_order=sorted(x1_order,key=lambda x1_order:x1_order[0])
    
    #Ahora ordenamos en 'x' de izquierda a derecha DE LOS DOS PUNTOS DE ABAJO
    x2_order=y_order[2:4]
    x2_order=sorted(x2_order,key=lambda x2_order:x2_order[0])
    
    #return los puntos ya ordenados
    return [x1_order[0],x1_order[1],x2_order[0],x2_order[1]]

image = cv2.imread('D:\ESCANER\img_00.jpeg')
#pasar a grises para encontrar las esquinas
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#metodo para dibujar bordes, el rango 10-150 es para dibujar solo lineas mas marcadas e ignorar las por debajo del min
canny = cv2.Canny(gray,10,150)
#engrosar lineas blancas detectadas por canny
canny = cv2.dilate(canny,None,iterations=1)

#Encontrar contornos externos. En openCV4 los contornos se almacenan en variable 0
cnts = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

#Seleccionar el contorno mas grande, o sea el de la imagen completa. 
#Lo hacemos ordenando de mayor a menor los contornos y seleccionando el primero que sería el más grueso
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:1]

#En este 'for' solo habrá un evento porque 'cnts' solo tiene un valor que es el contorno mas grueso
for c in cnts:
    #variable necesaria para PolyDP
    #Especifica la precision de aproximación a una forma definida (distancia máxima entre el contorno real y el contorno aproximado)
    #El primer valor es el porcentaje de precision, mientras mas pequeño, será mas apegado a la imagen real
    #True= curva cerrada, Flase=curva abierta
    epsilon = 0.01*cv2.arcLength(c,True)
    #funcion para detectar los contornos 
    #tendrá las coordenadas de todos los vértices encontrados
    approx = cv2.approxPolyDP(c,epsilon,True)
    
    #Cuando encuentre los 4 vértices de la imagen (porque es un triangulo o cuadrado), hay que dibujar el contorno
    if len(approx)==4:
        cv2.drawContours(image, [approx],0,(0,255,255),2)

        #llamar funcion 'ordenar_puntos' para ordenar las coordenadas de arriba abajo, izquierda a derecha
        puntos= ordenar_puntos(approx)

        #Dibujamos circulos en los vertices, puntos es la lista de coordenadas ordenadas de arriba a abajo, izq a dere
        cv2.circle(image, tuple(puntos[0]), 7, (255,0,0), 2)
        cv2.circle(image, tuple(puntos[1]), 7, (0,255,0), 2)
        cv2.circle(image, tuple(puntos[2]), 7, (0,0,255), 2)
        cv2.circle(image, tuple(puntos[3]), 7, (255,255,0), 2)

        #usamos la variable puntos porque ya están esritos en orden original
        pts1=np.float32(puntos)
        #puntos de la posicion deseada
        pts2=np.float32([[0,0],[270,0],[0,310],[270,310]])
        #matriz para transformacion
        M=cv2.getPerspectiveTransform(pts1,pts2)
        #transformacion
        dst=cv2.warpPerspective(gray,M,(270,310))
        cv2.imshow('dst',dst)

        texto = pytesseract.image_to_string(dst)
        print('texto: ',texto)


cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()