#!/usr/bin/python3

#Atualizacao 4 (por Michel Rodrigues)
#Programa mais completo
#conta, identifica e envia os registros para nuvem
#Diminuidos scaleFactor de 1.5 para 1.2
#Congela por fração de segundos a imagem para reconhecimento com maior qualidade
#Dados e status mostrados na tela
#Numero de contagem impresso na tela
#Removidos framecounter
#Adicionado estimativa de idade e sexo
#Area do corpo aumentada de 9000 para 15000 para melhorar detecção
#arrumado contador que não estava incrementando corretamente

import numpy as np
import cv2
import requests
import json
import time
import datetime
import imutils
import cv2 as cv
import math

file1 = open("contagem.txt", "r+")
file1.seek(37)
contagem = (file1.read(6))

ContadorSaidas = int(contagem)
#ContadorSaidas=0

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Masculino', 'Feminino']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream
#cap = cv.VideoCapture(args.input if args.input else 0)
j=0
def detect():
    # Read frame
    img = cv.imread("/home/pi/output.png")
    padding = 20
    frame = img
    
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        #print("Nao foi possivel estimar idade e sexo")
        #cv2.putText(frame1, "Nao foi possivel estimar idade e sexo", (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        j=0
        gender=None
        age=None
        file1 = open("contagem.txt", "w")
        
        L = ["Id: "+str(objectID)+" \n", "Flag: "+str(flag)+" \n", "Contagem:"  +str(ContadorSaidas)+"     \n", "\nNao foi possivel estimar idade e sexo \n"]
        file1.write("Registros: \n")
        file1.writelines(L)
        file1.close()
        #return 0
    z=len(bboxes)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        #print("Age Output : {}".format(agePreds))
        #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
        '''
        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.imshow("Teste_sexo_idade", frameFace)
        # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
        #print("time : {:.3f}".format(time.time() - t))
        '''
        j=1        
        #print(""+str(j)+" pessoa do sexo "+str(gender)+" com idade entre "+str(age)+"")
        while(j<z):
          j=j+1 
        #print(age , gender)
        #cv2.putText(frame1, ""+str(j)+" pessoa do sexo {} com idade entre {}".format(gender, age), (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        file1 = open("contagem.txt", "w")
        
    #salva registros txt
        L = ["Id: "+str(objectID)+" \n", "Flag: "+str(flag)+" \n", "Contagem:"  +str(ContadorSaidas)+"       \n", "\n"+str(j)+" pessoa do sexo "+str(gender)+" com idade entre "+str(age)+""]

        file1.write("Registros: \n")
        file1.writelines(L)
        file1.close()
        #return 1
    return z, j , gender, age       

#instancia uso da cam
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#obtem dimensoes
frame_largura = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_altura =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#variaveis globais, atualizadas no codigo
_x=0
_y=0
_h=0
_w=0
x=0
y=0
h=0
w=0

tir=0

#ContadorSaidas=0

ret, frame1 = cap.read()
ret, frame2 = cap.read()

OffsetLinhasRef = 50
attempts = 0
status = 400
a=None
b=0
c=0
d=0

flag=2

objectID=0
box=0
numFaces=0
faceOk=0

while cap.isOpened():
    
    CoordenadaYLinhaSaida = int(frame_altura / 2)+OffsetLinhasRef
    CoordenadaYLinhaEntrada = int(frame_altura / 2)-OffsetLinhasRef
    #
     #adicionar uma linha de segurança ou timeout
    #
    #CoordenadaFlag = int(frame_largura / 2)
    #frame = imutils.resize(frame1, width=400)
    #cv2.imshow("fe2ed", frame)
    '''
    #Tentativa de localizar rosto, se ok, compara com os contornos de predicao
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2)
    for (_x,_y,_w,_h) in faces:
               
          color = (255,0,0) #BGR
          stroke = 2
          end_cord_x = _x + _w
          end_cord_y = _y + _h
          cv2.rectangle(frame1, (_x,_y), (end_cord_x, end_cord_y), color, stroke)
          
    faces_cont=len(faces)
    #print(faces_cont)
    '''
    #Calcula diferenca entre frames, eliminando o tracker
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    _,contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #contornos de predicao de pessoas
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < 10000 :
            
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        if(y > CoordenadaYLinhaEntrada):
            continue
        
        gray = cv2.cvtColor(frame1.copy(), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        faces_cont=len(faces)
        
        while(numFaces < faces_cont):
            numFaces+=1
        
        for (_x,_y,_w,_h) in faces:
               
          color = (255,0,0) #BGR
          stroke = 1
          end_cord_x = _x + _w
          end_cord_y = _y + _h
          cv2.rectangle(frame1, (_x,_y), (end_cord_x, end_cord_y), color, stroke)
        
        #salva imagem na melhor vez,quando surge
        if(faces_cont >=1 and numFaces == faces_cont):
          faceOk=1
          img_gray=frame1[y:(frame_altura-200), x:x+w]
          img_name = "/home/pi/output.png"
          cv2.imwrite(img_name, img_gray)
        '''  
        #nao utilizado por enquanto
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
        for (_x,_y,_w,_h) in faces:
          
          img_gray=gray[y:y+h, x:x+w]
          img_name = "/home/pi/myimage.png"
          cv2.imwrite(img_name, img_gray)
          
          color = (255,0,0) #BGR
          stroke = 2
          end_cord_x = _x + _w
          end_cord_y = _y + _h
          cv2.rectangle(frame1, (_x,_y), (end_cord_x, end_cord_y), color, stroke)
        a=len(faces)
        print(a)
        '''
    
    #print(y,_y)
    
    #testa flags, operadores and garantem a predicao 
    #adicionado contador de boxes nas pessoas    
    if( (faceOk == 1) and
        (y < CoordenadaYLinhaEntrada)):
      flag=0
      attempts=0
      box=numFaces
      cv2.putText(frame1, "Flag atualizada", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) 
      '''
      img_gray=frame1[0:frame_altura, 0:frame_largura]
      #img_name = "myimage"+str(ContadorSaidas)+".png"
      img_name = "myimage.png"
      cv2.imwrite(img_name, img_gray)
      '''
    
 
    if((y > CoordenadaYLinhaSaida) and
       (flag ==0)):
      cv2.putText(frame1, "Flag setada", (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  
      faceOk=0 
      flag=1
      ContadorSaidas = ContadorSaidas + numFaces
      contours=[]
      faces_cont=0
      numFaces=0
      box=0
      (a,b,c,d) = detect()
    #print(a,b,c,d)
    if(a != None and a>0):
      cv2.putText(frame1, ""+str(b)+" pessoa do sexo {} com idade entre {}".format(c, d), (5, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    if (a == 0):
      cv2.putText(frame1, "Nao foi possivel estimar o genero e a idade", (5, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

      
    #print("pessoas na imagem:")
    #print(box)
    #cv2.putText(frame1, "Flag entrada: {}".format(str(flag)), (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame1, "Contagem: {}".format(str(ContadorSaidas)), (5, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame1, "Pessoas detectadas: {}".format(str(numFaces)), (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 55), 2)
    #cv2.putText(frame1, "dim: {} {} {} {} ".format(str(x),str(y),str(x+w),str(y+h)), (5, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 55), 2)

    #redimensionamento de frame nao utilizado por enquanto
    
    #frame1 = cv2.resize(frame1, (400,300))
    #out.write(image)
    
    cv2.line(frame1, (0,CoordenadaYLinhaSaida), (frame_largura,CoordenadaYLinhaSaida), (120, 0, 0), 1)
    cv2.putText(frame1, "Pressione ESC para sair", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    
    file1 = open("contagem.txt", "w")
    #salva registros txt
    L = ["Id: "+str(objectID)+" \n", "Flag: "+str(flag)+" \n", "Contagem:"  +str(ContadorSaidas)+" \n"]

    file1.write("Registros: \n")
    file1.writelines(L)
    file1.close()
    print(CoordenadaYLinhaSaida , y)
    #se o token eh valido, envia o registro
    if (flag == 1 and attempts == 0):
        
      if ( status >= 400 and attempts <=5):
            attempts +=1
            timestamp = datetime.datetime.now()
            url = 'http://facesapi.trudata.com.br/api/v1/Login'
            payload = {
                      'userName': 'jedi@hansenautomacao.com',
                      'password': '@H4ns3n!.'
                      }
            headers = {'content-type': 'application/json'}
            r = requests.post(url, data=json.dumps(payload), headers=headers)
            r.status_code
            status = r.status_code
      
            #print(status)
            #print(r.text)

            response = r.json()
      
            token=response.get('accessToken')
      else:
            attempts=1
            url1 = 'http://facesapi.trudata.com.br/api/v1/ReconData/Single'
            ts = timestamp.strftime("%m/%d/%Y %H:%M:%S")
            payload = {
                      "codigo": ContadorSaidas,
                      "cdEmpresa": 2,
                      "cdLocal": 1,
                      "raspId": "ef9f40de-cf83-48f5-b360-e5d269bbe71a",
                      "idadeAprox": 23,
                      "genero": "M",
                      "horarioRecon": ""+str(ts)+""
                      }
            head = {'Authorization': 'Bearer ' + token,'content-type': 'application/json'}
            recon = requests.post(url=url1, data=json.dumps(payload), headers=head)
            Tstatus = recon.status_code
            #print(Tstatus)
            if (Tstatus > 200):
              cv2.putText(frame1, "Erro no envio", (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
              time.sleep(5)
              break
            else:
              cv2.putText(frame1, "Registro enviado para a nuvem!...", (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
           
    #se tudo ok, mostra a imagem
    #imprime textos na tela
    cv2.imshow("Output", frame1)

    #atualiza o frame
    frame1 = frame2
    ret, frame2 = cap.read()
    
    #fecha a tela
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
