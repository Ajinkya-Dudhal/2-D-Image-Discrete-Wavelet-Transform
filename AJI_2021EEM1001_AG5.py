#2D-DWT code
#Ajinkya Dudhal_2021EEM1001_Assigment-5
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pywt

#D:/IIT Ropar/Sem-1_Stuff/EE525-Communication and Signal Processing Lab/Assignment-3/sample_image.jpeg
n=int(input("Enter '1' for logic and '2' for in-built function: "))
img=mpimg.imread(input("Enter file path: "))
level=int(input("Enter number of decomposition levels="))
R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
print(np.shape(imgGray))

if n==1:
    #Rowwise
    for k in range(level):
        num_rows=np.shape(imgGray)[0]
        LP0=[]
        HP0=[]
        for i in range(num_rows):
            x=np.convolve(imgGray[i,:],[1/np.sqrt(2),1/np.sqrt(2)])
            LP0.append(x)
            y=np.convolve(imgGray[i,:],[1/np.sqrt(2),-1/np.sqrt(2)])
            HP0.append(y)
        down_LP0=np.array(LP0)
        down_HP0=np.array(HP0)
        down_LP0=down_LP0[:,::2]
        down_HP0=down_HP0[:,::2]

        #Columnwise Low Pass
        num_columns=np.shape(down_LP0)[1]
        LP1=[]
        HP1=[]
        for i in range(num_columns):
            x=np.convolve(down_LP0[:,i],[1/np.sqrt(2),1/np.sqrt(2)])
            LP1.append(x)
            y=np.convolve(down_LP0[:,i],[1/np.sqrt(2),-1/np.sqrt(2)])
            HP1.append(y)
        cA1=np.array(LP1)
        cH1=np.array(HP1)
        if level%2!=0:
            cA1=np.transpose(cA1)
            cH1=np.transpose(cH1)
        cA1=cA1[::2,:]
        cH1=cH1[::2,:]

        imgGray=cA1

        #Columnwise High Pass
        num_columns=np.shape(down_HP0)[1]
        LP1=[]
        HP1=[]
        for i in range(num_columns):
            x=np.convolve(down_HP0[:,i],[1/np.sqrt(2),1/np.sqrt(2)])
            LP1.append(x)
            y=np.convolve(down_HP0[:,i],[1/np.sqrt(2),-1/np.sqrt(2)])
            HP1.append(y)
        cV1=np.array(LP1)
        cD1=np.array(HP1)
        if level%2!=0:
            cV1=np.transpose(cV1)
            cD1=np.transpose(cD1)
        cV1=cV1[::2,:]
        cD1=cD1[::2,:]

        if level%2==0:
            temp=cH1
            cH1=cV1
            cV1=temp
    plt.suptitle('Using Logic',fontsize='30')

else:
    coefficients=pywt.wavedec2(imgGray,'haar',level=level)
    cA1=coefficients[0]
    cH1,cV1,cD1=coefficients[1]
    plt.suptitle('Using Logic',fontsize='30')

plt.subplot(2,2,1)
plt.title("cA{}:Approximation Coefficient".format(level))
plt.imshow(cA1,cmap="gray")
plt.subplot(2,2,2)
plt.title("cH{}:Horizontal Coefficient".format(level))
plt.imshow(cH1,cmap="gray")
plt.subplot(2,2,3)
plt.title("cV{}:Vertical Coefficient".format(level))
plt.imshow(cV1,cmap="gray")
plt.subplot(2,2,4)
plt.title("cD{}:Diagonal Coefficient".format(level))
plt.imshow(cD1,cmap="gray")
plt.show()