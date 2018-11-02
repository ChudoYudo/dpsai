import random
from PIL import Image, ImageDraw #Подключим необходимые библиотеки.
import matplotlib.pyplot as plt
from collections import Counter
import pandas
import operator
import numpy as np
import sys
from copy import deepcopy
import math

class Centre:
    Number=0
    x=0
    y=0
    z=0
    __points_include=[]#<------------------------------------------------------------ вот это поле
    def constr(self,xx=0,yy=0,zz=0,nNumber=0):
        self.Number=nNumber
        self.y=yy
        self.x=xx
        self.z=zz
    def add_p(self,P):
        self.__points_include.append(P)
    def set_x(self,xx):
        self.x=xx
    def set_y(self,yy):
        self.y=yy
    def set_Number(self,nn):
        self.Number=nn
    def print_incl(self):
        print (self.__points_include)
    def distance_colculate(self,Object):
        return math.sqrt(math.pow(self.y-Object.get_sq(),2)+math.pow(self.x-Object.get_ar(),2))

def adder_point(C,P):
    C.add_p(P)

def includer(Centres,Objects):
    for point in Objects:
        Centres
        min = Centres[0].distance_colculate(point)
        num=0
        for centr in Centres:
            distance=centr.distance_colculate(point)
            if (distance<min):
                min=distance
                num=centr.Number
        adder_point(Centres[num],point)#<----------------------------------------------------------------- вот я пытаюсь писать в разные объекты

class Obj:
    Name=0
    Sqare=0
    Area=0
    Point_max=[]
    Point_min=[]
    def constr(self,N,S,A):
        self.Name=N
        self.Area= A
        self.Sqare = S
    # def elongation_colculate(mass):

    def shw(self):
        print("Name :"+str(self.Name))
        print("Sq :" + str(self.Sqare))
        print("Area :" + str(self.Area))
    def mass_centre(Objects):
        return (sum(pt.Sqare for pt in Objects) / len(Objects),
              sum(pt.Area for pt in Objects) / len(Objects))

    def set_sq(self,S):
        self.Sqare=S
    def set_ar(self,A):
        self.Area=A
    def get_sq(self):
        return self.Sqare
    def get_ar(self):
        return self.Area

sys.setrecursionlimit(15000000)
core=[1,0,-1,
      2,0,-2,
      1,0,-1]


def show_plt(aa):
    aaa = Counter(aa)
    sorted_x = sorted(aaa.items(), key=operator.itemgetter(0))
    df = pandas.DataFrame.from_dict(dict(sorted_x), orient='index')
    df.plot(kind='bar')
    #plt.show()

def gray(image):
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    aa=[]

    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            S = 0
            a = pix[i, j][0]
            if (a!=0):
                S=255
            aa.append(S//10)
            draw2.point((i, j), (S, S, S))
    img_fin.save("gray.jpg", "JPEG")
    show_plt(aa)

def gray_shades(image):
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    aa=[]
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0]
            b = pix[i, j][1]
            c = pix[i, j][2]
            S = (int)(a*0.3 + b*0.59 + c*0.11)
            aa.append(S//10)
            draw2.point((i, j), (S, S, S))
    img_fin.save("shades.jpg", "JPEG")
    show_plt(aa)
def negativ(image):
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    aa=[]
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0]
            b = pix[i, j][1]
            c = pix[i, j][2]
            S = (int)((255-a) + (255-b) + (255-c))
            aa.append(S//10)
            draw2.point((i, j), (S, S, S))
    img_fin.save("negative.jpg", "JPEG")
    show_plt(aa)
def binary_image(image,gmax):

    S=28
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    aa=[]
    matrix=np.zeros((width,height))
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0]
            if(a<gmax):
                S=0
                matrix[i][j] = 1
            if(a>gmax-1):
                S=255
                matrix[i][j]=0

            if(S==28):
                print("krfmgjeorkgjeorg")
            aa.append(S)
            draw2.point((i, j), (S, S, S))
    img_fin.save("binary.jpg", "JPEG")
    show_plt(aa)
    return matrix

def core_filt(image,core):
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  #
    bb=[]
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            sum=0
            mas_pix = []
            kkk=pix[i,j][0]
            sum=sum+(kkk*core[4])
            if (i!=width-1 and j!=height-1):
                sum=sum+(pix[i+1,j+1][0]*core[8])
            if (i!=0 and j!=0):
                sum = sum + (pix[i - 1, j - 1][0] * core[0])
            if (i!=0 and j!=height-1):
                sum = sum + (pix[i - 1, j + 1][0] * core[6])
            if (i!=width-1 and j!=0):
                sum = sum + (pix[i + 1, j - 1][0] * core[2])
            if (j!=height-1):
                sum = sum + (pix[i, j + 1][0] * core[7])
            if (j!=0):
                sum = sum + (pix[i , j - 1][0] * core[1])
            if (i!=width-1):
                sum = sum + (pix[i + 1, j ][0] * core[5])
            if (i!=0):
                sum = sum + (pix[i - 1, j ][0] * core[3])
            if( mas_pix==0 ):
                continue

            n= sum
            bb.append(n//100)
            draw2.point((i, j), (n,n,n))
    img_fin.save("core_filtre.jpg", "JPEG")
    show_plt(bb)
def median_filtre(image):
    pix=image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    bb=[]
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            mas_pix = []
            kkk=pix[i,j][0]
            mas_pix.append(pix[i,j][0])
            if (i!=width-1 and j!=height-1):
                mas_pix.append(pix[i+1,j+1][0])
            if (i!=0 and j!=0):
                mas_pix.append(pix[i-1, j-1][0])
            if (i!=0 and j!=height-1):
                mas_pix.append(pix[i-1, j+1][0])
            if (i!=width-1 and j!=0):
                mas_pix.append(pix[i+1, j-1][0])
            if (j!=height-1):
                mas_pix.append(pix[i, j+1][0])
            if (j!=0):
                mas_pix.append(pix[i, j-1][0])
            if (i!=width-1):
                mas_pix.append(pix[i+1, j][0])
            if (i!=0):
                mas_pix.append(pix[i-1, j][0])
            mas_pix.sort()
            n= mas_pix[len(mas_pix)//2]
            bb.append(n//10)
            draw2.point((i, j), (n,n,n))
    img_fin.save("median_filtre.jpg", "JPEG")
    show_plt(bb)

def min_max_filtre(image,min_or_max):
    pix=image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    bb=[]
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            mas_pix = []
            kkk=pix[i,j][0]
            mas_pix.append(pix[i,j][0])
            if (i!=width-1 and j!=height-1):
                mas_pix.append(pix[i+1,j+1][0])
            if (i!=0 and j!=0):
                mas_pix.append(pix[i-1, j-1][0])
            if (i!=0 and j!=height-1):
                mas_pix.append(pix[i-1, j+1][0])
            if (i!=width-1 and j!=0):
                mas_pix.append(pix[i+1, j-1][0])
            if (j!=height-1):
                mas_pix.append(pix[i, j+1][0])
            if (j!=0):
                mas_pix.append(pix[i, j-1][0])
            if (i!=width-1):
                mas_pix.append(pix[i+1, j][0])
            if (i!=0):
                mas_pix.append(pix[i-1, j][0])
            if (min_or_max==1):
                n=min(mas_pix)
            if (min_or_max==0):
                n=max(mas_pix)
            bb.append(n//10)
            draw2.point((i, j), (n,n,n))
    if (min_or_max == 1):
        img_fin.save("min_filtre.jpg", "JPEG")
    if (min_or_max == 0):
        img_fin.save("max_filtre.jpg", "JPEG")
    show_plt(bb)


def brightness_corrector(image):
    pix = image.load()
    width = image.size[0]  # Определяем ширину.
    height = image.size[1]  # Определяем высоту.
    factor = 70
    img_fin = Image.new('RGB', (width, height))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(width):
        for j in range(height):
            a = pix[i, j][0] + factor
            b = pix[i, j][1] + factor
            c = pix[i, j][2] + factor
            draw2.point((i, j), (a, b, c))
    img_fin.save("bri.jpg", "JPEG")

# gray_shades(pix)
#
# image2 = Image.open("ans1.jpg")
#
# median_filtre(image2)
# binary_image(image2)
def razm(image,m,n):
    km=0
    kn=0
    cur=1
    for i in range(m):
        for j in range (n):
            kn=j-1
            if (kn<=0):
                kn=1
                B=0
            else:
                B=image[i,kn]
            km=i-1
            if (km<=0):
                km=1
                C=0
            else:
                C=image[i,j]
            A=image[i,j]
            if (A==0):
                kk=0
            elif (B==0 and C==0):
                cur=cur+1
                image[i,j]=cur
            elif (B!=0 and C==0):
                image[i,j]=B
            elif(B==0 and C!=0):
                image[i,j]=C
            elif(B!=0 and C!=0):
                if (B==C):
                    image[i,j]=B
                else:
                    image[i,j]=B

def labeling (img,labels,W,H):
    L=1
    for y in range(H):
        # print('----')
        for x in range(W):
            L=L+1
            Fill(img,labels,x,y,L-1,W,H)
            # print (x)
            # print(y)

def Fill(img,labels,x,y,L,W,H):
    if (labels[x][y]==0 and img[x][y]==1):
        labels[x][y]=L
        if(x>0):
            Fill(img,labels,x-1,y,L,W,H)
        if (x < W-1):
            Fill(img, labels, x + 1, y, L,W,H)
        if (y > 0):
            Fill(img, labels, x, y-1, L,W,H)
        if (y<H-1):
            Fill(img, labels, x, y+1, L,W,H)


def dr(matrix,M,N,k):
    img_fin = Image.new('RGB', (M, N))
    draw2 = ImageDraw.Draw(img_fin)
    for i in range(M):
        for j in range(N):
            if(matrix[i][j]!=0):
                draw2.point((i, j), (0, 0, 0))
            else:
                draw2.point((i, j), (255, 255, 255))
    if (k==1):
        img_fin.save("1.jpg", "JPEG")
    else:
        img_fin.save("2.jpg", "JPEG")


def drr(matrix,M,N):
    img_fin = Image.new('RGB', (M, N))
    draw2 = ImageDraw.Draw(img_fin)
    aa=[]
    for i in range(M):
        for j in range(N):
            if(matrix[i][j]==0):
                draw2.point((i, j), (255, 255, 255))
            else:
                aa.append(matrix[i][j])
                draw2.point((i, j), (int(matrix[i][j]), int(matrix[i][j]*2), int(mat[i][j]//3)))

        #img_fin.save("2.jpg", "JPEG")
    aaa = Counter(aa)
    return dict(aaa)

def renumber(dictt):
    ret=[[]]
    l=1
    for k in dictt.keys():
        dictt[l]=dictt.pop(k)
        l+=1
    print(dictt)


def perim(matr,W,H,listt):
    k=0
    img_fin = Image.new('RGB', (W, H))
    draw2 = ImageDraw.Draw(img_fin)

    countt=np.zeros(len(listt))
    for y in range(W):
        for x in range(H):
            try:
                bl=((matr[y][x]!=0)and (matr[y+1][x]==0 or matr[y-1][x]==0 or matr[y][x+1]==0 or matr[y][x-1]==0))
            except:
                continue
            if (bl):
                lol=listt.index(matr[y][x])
                countt[lol]+=1
                draw2.point((y, x), (255, 255, 255))

    z=dict(zip(listt,countt))

    img_fin.save("11.jpg", "JPEG")
    return z


def kme ():
    centers_old = np.zeros(centers.shape)  # to store old centers
    centers_new = deepcopy(centers)  # Store new centers

    data.shape
    clusters = np.zeros(n)
    distances = np.zeros((n, k))

    error = np.linalg.norm(centers_new - centers_old)

    # When, after an update, the estimate of that center stays the same, exit loop
    while error != 0:
        # Measure the distance to every center
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        # Calculate mean for every cluster and update the center
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(centers_new - centers_old)
    centers_new


# mat_m=5
# mat_n=5
# kk=[1,2]
# matrix=[[0,0,0,0,0],[0,1,1,0,0],[0,1,0,0,0],[0,0,2,2,0],[0,0,0,0,0]]
# perim(matrix,5,5,kk)
# labels=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
# #razm(matrix,5,5)
# dr(matrix,5,5,1)
# for i in range(4):
#     print(matrix[i])
# labeling(matrix,labels,4,4)
# dr (labels,5,5,0)
# print("-----------------------------------------------------------")
# for i in range(4):
#     print(labels[i])


kk=[]
Cent=Centre()
Cent.set_x(3)
Cent.set_y(2)
Ob=Obj()
Ob.set_ar(3)
Ob.set_sq(7)

Ob1=Obj()
Ob1.set_ar(7)
Ob1.set_sq(7)

Ob2=Obj()
Ob2.set_ar(3)
Ob2.set_sq(11)

Ob3=Obj()
Ob3.set_ar(7)
Ob3.set_sq(11)

ss=[]
Cen1=Centre()
Cen1.set_y(4)
Cen1.set_x(5)
Cen1.set_Number(0)


Cen2=Centre()
Cen2.set_y(13)
Cen2.set_x(5)
Cen2.set_Number(1)

ss.append(Cen1)
ss.append(Cen2)


kk.append(Ob)
kk.append(Ob1)
kk.append(Ob2)
kk.append(Ob3)

K=includer(ss,kk)

for s in ss :
    s.print_incl()

# --------------------------------------------------------------------------------------------------------------------------------------
# image3 = Image.open("neg11.jpg")
# width = image3.size[0]  # Определяем ширину.
# height = image3.size[1]
#
# lab=np.zeros((width,height))
#
#
# gray_shades(image3)
#
#
#
# image3 = Image.open("shades.jpg")
# mat=binary_image(image3,180)
#
# image3 = Image.open("binary.jpg")
# negativ(image3)
#
# image3 = Image.open("negative.jpg")
# min_max_filtre(image3,0)
#
# image3 = Image.open("max_filtre.jpg")
# min_max_filtre(image3,1)
#
# image3 = Image.open("min_filtre.jpg")
# min_max_filtre(image3,1)
#
#
# image3 = Image.open("min_filtre.jpg")
# mat=binary_image(image3,127)
#
# dr(mat,width,height,1)
# labeling(mat,lab,width,height)
#
#
# kek=drr(lab,width,height)
#
#
#
# ts=[]
# for keks in kek:
#     A=Obj()
#     A.constr(keks,kek.get(keks),0)
#     ts.append(A)
#
# print(len(kek))
# lol=perim(lab,width-2,height-2,list(kek.keys()))
# print(lol)
#
# for t in ts:
#     t.set_ar(lol.get(t.Name))
#
# for t in ts:
#     print("------------")
#     t.shw()
#
# zz=[[]]
# z=dict(zip(list(kek.values()), list(lol.values())))
# print (type(z))
#
#
# print(lab)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# plt.plot(list(kek.values()), list(lol.values()), 'ro')
# plt.axis([0,4000,0,4000])


# plt.show()




#===========================================================================================
# center_1 = np.array([9,1])
# center_2 = np.array([8,2])
# center_2 = np.array([8,2])
# # Generate random data and center it to the three centers
# data_1 = [[9,1],[8,2]]
# data_2 = [[7,1],[1,1]]
# data_3 = [[1,3],[2,2]]
# data_4 = [[2,1],[1,2]]
#
# # lsp=[]
# # for t in ts:
# #     lsp.append([t.Area,t.Sqare,1,1])
# det=np.array([[1,4,1],[3,4,1],[8,9,1],[4,1,1],[9,6,1],[6,8,1],[8,1,1],[9,1,1],[8,2,1]])
#
#
# # det=np.array(lsp)
# print(det)
# data = det
# plt.scatter(data[:,0], data[:,1], s=7)
#
#
# k = 3
# # Number of training data
# n = data.shape[0]
# # Number of features in the data
# c = data.shape[1]
#
# # Generate random centers, here we use sigma and mean to ensure it represent the whole data
# mean = np.mean(data, axis = 0)
# std = np.std(data, axis = 0)
# centers = np.random.randn(k,c)*std + mean
#
# # Plot the data and the centers generated as random
# plt.scatter(data[:,0], data[:,1], s=7)
# plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
# plt.show()
#
# centers_old = np.zeros((centers.shape))  # to store old centers
# centers_new = deepcopy(centers)  # Store new centers
#
# data.shape
# clusters = np.zeros(n)
# distances = np.zeros((n, k))
#
# error = np.linalg.norm(centers_new - centers_old)
#
# # When, after an update, the estimate of that center stays the same, exit loop
# while error != 0:
#     # Measure the distance to every center
#     for i in range(k):
#         distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
#     # Assign all training data to closest center
#     clusters = np.argmin(distances, axis=1)
#
#     centers_old = deepcopy(centers_new)
#     # Calculate mean for every cluster and update the center
#     for i in range(k):
#         centers_new[i] = np.mean(data[clusters == i], axis=0)
#     error = np.linalg.norm(centers_new - centers_old)
# centers_new
#
# plt.scatter(data[:,0], data[:,1], s=7)
# plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
# plt.show()
# print (centers_new)
#=================================================================================================










# renumber(kek)


#
# image3=Image.open("bolt.jpg")
# median_filtre(image3)
#

#
# image3=Image.open("binary.jpg")
# median_filtre(image3)
#
# image3 = Image.open("median_filtre.jpg")
# binary_image(image3,30)
#
# image3 = Image.open("binary.jpg")
# gray(image3)


# image.save("ans.jpg", "JPEG")
# for i in range(width):
#     for j in range(height):
#         a = pix[i, j][0]
#         b = pix[i, j][1]
#         c = pix[i, j][2]
#         aa.append(S//10)
#         draw.point((i, j), (255-a, 255-b, 255-c))
# image.save("ans2.jpg", "JPEG")
# plt.hist(aa, bins=50)
# plt.savefig('hist.png')
# bb=Counter(aa)
# df = pandas.DataFrame.from_dict(bb, orient='index')
# df.plot(kind='bar')
# plt.show()



# class Obj:
#     Name=0
#     Sqare=0
#     Area=0
#     def constr(self,N,S,A):
#         self.Name=N
#         self.Area= A
#         self.Sqare = S
#
#     def shw(self):
#         print("Name :"+str(self.Name))
#         print("Sq :" + str(self.Sqare))
#         print("Area :" + str(self.Area))


# A=Obj()
# B=Obj()
#
# A.constr(1,2,3)
# B.constr(4,5,6)
# lis=[]
# lis.append(A)
# lis.append(B)
# for li in lis:
#     li.shw()