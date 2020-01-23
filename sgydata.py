import struct
import numpy as np
import random
import math
def norm_sgy(data):
    maxval=max(max(data))
    minval=min(min(data))
    return [[(float(i)-minval)/(float(maxval-minval)+0.01e-100) for i in j] for j in data]
def norm_sgy1(data):
    maxval=max([max([abs(i) for i in j]) for j in data])
    #minval=min(min(data))
    #return [[(float(i)-min(j))/float(max(j)-min(j)) for i in j] for j in data]
    #return [[float(i)/(float(max(j))+0.01e-100) for i in j] for j in data]
    return [[float(i)/(maxval+0.01e-100) for i in j] for j in data]
def read_sgy(sgynam='test.sgy'):
#    print "sgynam: "+sgynam
    try:
        binsgy = open(sgynam,'rb')
    except IOError:
        return 0,0,[]
    fhead=binsgy.read(3600);
#    print fhead[3213:3215]
    nr=struct.unpack(">H",fhead[3212:3214])
    print(nr)
    nsmp=struct.unpack(">H",fhead[3220:3222])
    print(nsmp)
    data = []
    for ir in range(0,nr[0]):
      trchead=binsgy.read(240)
      trcdata=binsgy.read(nsmp[0]*4)
      data1 = []
      for i in range(0,nsmp[0]):
       # print(trcdata[i*4:i*4+4])
        data1=data1+list(struct.unpack(">f",trcdata[i*4:i*4+4]))
      data.append(data1)
    print("read 1sgy end")
    binsgy.close()
    return nr,nsmp,data;

def loca_img_xyz(xr=[0.25,0.01,24],yr=[-0.2,0.013,32],zr=[3.07,0.01,18],xyz=[0.4,0.0,3.12],r=0.0005,rtz=(100.0/12.0)**2):
    img=[]
    #rtz=(100.0/12.0)**2;
    #rtz=1.0;
    for i in range(0,xr[2]):
       x = xr[0]+xr[1]*i
       tmp1=[]
       for j in range(0,yr[2]):
          y=yr[0]+yr[1]*j
          tmp2=[]
          for k in range(0,zr[2]):
             z=zr[0]+zr[1]*k
             ftmp=(x-xyz[0])*(x-xyz[0])+(y-xyz[1])*(y-xyz[1])+rtz*(z-xyz[2])*(z-xyz[2])
             tmp2=tmp2+[math.exp(-0.5*ftmp/r)]
          tmp1.append(tmp2)
       img.append(tmp1);
    return img;

def shuffle_data(data,ydata,seed,shuffle):
    if shuffle == 'false':
        return data,ydata;
    index=[i for i in range(len(ydata))]
    random.seed(seed)
    random.shuffle(index)
    data = [data[i] for i in index]
    ydata = [ydata[i] for i in index]
    return data,ydata
    
def load_sgylist_xyz1(sgylist=['./path/','./ok_syn1/sgylist.txt'],sgyr=[0,-1,1],xr=[3913.880-25.0,14.000,20],yr=[-10896.620-25,15.000,20],zr=[100001.000-3,3.00,10],r=500.000,rtz=(100.0/12.0)**2,
                      shuffle='true',shiftdata=[list(range(-5,2)),1]):
    #nx,ny,stn=read_stn(stnnam)
    with open(sgylist[1],'r') as f:
        lines=f.readlines()
    lines=lines[sgyr[0]:sgyr[1]:sgyr[2]]+[lines[sgyr[1]]];
    data= []
    ydata=[]
    #eventnam=[]
    for i in range(0,len(lines)):
       line1=lines[i].split()
       sgynam=sgylist[0]+line1[0].strip()
       loca=[float(num) for num in line1[1:4]]
       img=loca_img_xyz(xyz=[loca[0],loca[1],loca[2]],xr=xr,yr=yr,zr=zr,r=r,rtz=rtz)
       print(sgynam)
       nr,nsmp,data1 = read_sgy(sgynam);
       data1=np.clip(np.nan_to_num(np.array(data1)),-1.0e-2,1.0e-2).tolist()
#       data1 = prep_data(data1,stn,nx,ny)
       if nr != 0:
         data1=norm_sgy1(data1)
         data1=[[[data1[ir][j],data1[ir+1][j],data1[ir+2][j]] for j in range(nsmp[0])] for ir in range(0,nr[0],3)]
         data.append(data1)
         ydata.append(img);
         #eventnam.append(sgynam)
       else:
         print('1 event sgy not found')
    if shiftdata[1]>0:
        data1,ydata1=augment_data2(data=data,ydata=ydata,shiftdata=shiftdata);
        data=data+data1;
        ydata=ydata+ydata1;

    data,ydata=shuffle_data(data,ydata,1,shuffle);
    data=np.array(data)
    ydata=np.array(ydata)
    return data,ydata
   
def augment_data2(data=[],ydata=[],shiftdata=[list(range(-5,2)),1]):
   #     data_out,ydata_out = cut_trace(icut=par[0],data=data,ydata=ydata);
        data1=[];
        ydata1=[];
        nsmp=len(data[0][0])
        for i in range(len(ydata)):
            random.seed(i);
            its=random.sample(shiftdata[0],shiftdata[1]);
            for j in range(0,len(its)):
                if its[j]<0:
                    data_tmp=[ftmp[nsmp+its[j]:]+ftmp[0:nsmp+its[j]] for ftmp in data[i]]
                else:
                    data_tmp=[ftmp[its[j]:]+ftmp[0:its[j]] for ftmp in data[i]]
                ydata_tmp=ydata[i];
                data1=data1+[data_tmp];
                ydata1=ydata1+[ydata_tmp];
        return data1,ydata1;

if __name__ == '__main__':
#     nr,nsmp,data=read_sgy()
#     print nr,nsmp,data[1:nsmp[0]]
     # numsgy,data,ydata=load_sgylist()
     # print numsgy,len1,len(data),data[1],data.shape,ydata.shape
     # img=loca_img()
     # print ydata;
     # print('mask', ydata[0][10])
     import scipy.io as sio
     # print(read_stn('ok_syn1/ok.stn'))
     # nam,data,ydata=load_sgylist_xyz1()
     nams,data,ydata=load_sgylist_xyz1(sgylist='loca_ok2_sgylist_largercv_train_n1013.txt',shuffle='false',
            sgyr=[0,5,1],xr=[3913.880-25.0,3.500,80],yr=[-10896.620-45,2.500,128],zr=[0.000,0.4,30],r=400,twin=list(range(100,800)),asize=0,
            shiftdata=[list(range(20,50))+list(range(-200,-20)),0],doubleevent=[list(range(-200,-150)),1])
     #print(data,data.shape)[list(range(20,50))+list(range(-200,-20)),1]
     sio.savemat('D:/cnnloca/test.mat',{'data':data})


