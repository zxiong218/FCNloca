%%
clear all
load('../test_predictedlabel.mat');ydata=double(loca_predict);
%load('\\vmware-host\Shared Folders\vmshare1\Italy_earthquake\round1\result_trbfm6.5_test9samp\xydata_pt420.mat');ydata=double(imgloca);
%load('D:\project\italy\test\multi_win\model_t220-t1200\new_edit\t260\test1.mat')

ie=1;
%xr1=[-2667.0,0.5,96];
%yr1=[-4145.0,0.5,192];
%zr1=[0.0,0.5,64];
xr1=xyzrange(1,:);
yr1=xyzrange(2,:);
zr1=xyzrange(3,:);

img=reshape(ydata(ie,:,:,:),xr1(3),yr1(3),zr1(3));
sz=size(img);
tmp=0;
for i=1:sz(1)
    for j=1:sz(2)
        for k=1:sz(3)
            if(tmp<img(i,j,k))
                tmp=img(i,j,k);
                tmpi=i;
                tmpj=j;
                tmpk=k;
            end
        end
    end
end
imgxy=reshape(img(:,:,tmpk),xr1(3),yr1(3));
imgxz=reshape(img(:,tmpj,:),xr1(3),zr1(3));
tmpy=(tmpj-1)*yr1(2)+yr1(1);
tmpx=(tmpi-1)*xr1(2)+xr1(1);
tmpz=(tmpk-1)*zr1(2)+zr1(1);
xr=xr1(1):xr1(2):((xr1(3)-1)*xr1(2))+xr1(1);
yr=yr1(1):yr1(2):((yr1(3)-1)*yr1(2))+yr1(1);
zr=zr1(1):zr1(2):((zr1(3)-1)*zr1(2))+zr1(1);

[xx,yy,zz]=meshgrid(xr,tmpy,zr);
figure;
ha = tight_subplot(2,2,[.07 .07],[.08 .05],[.08 .05])
axes(ha(4))
surf(squeeze(xx),squeeze(yy),squeeze(zz),squeeze(img(:,tmpj,:)));
hold on;
[xx,yy,zz]=meshgrid(tmpx,yr,zr);
surf(squeeze(xx),squeeze(yy),squeeze(zz),squeeze(img(tmpi,:,:)));
[xx,yy,zz]=meshgrid(xr,yr,tmpz);
surf(squeeze(xx),squeeze(yy),squeeze(zz),squeeze(img(:,:,tmpk)'));
shading interp;
axis tight
view([-37.5,50])

colorbar
caxis([0,1]);


ylabel('X ');xlabel('Y ');zlabel('Depth (km)')

axes(ha(3))
imagesc(xr,zr,imgxz');hold on
caxis([0,1]);
plot(tmpx,tmpz,'pentagram','color','k','markersize',20,'markerface','k')
ylabel('Depth ');xlabel('X ');

axes(ha(2))
imagesc(xr,yr,imgxy');hold on%colorbar;
caxis([0,1]);
plot(tmpx,tmpy,'pentagram','color','k','markersize',20,'markerface','k')
ylabel('X ');xlabel('Y ');

nr=30;
nsmp=2048;
t=1:nsmp;
e1=reshape(wave_test(ie,:,:,3),nr,nsmp);
e1=e1/max(max(e1));
offset=max(max(e1));
scale=1;
axes(ha(1))
plot_nr_trc(t,e1,scale)
set(gcf,'unit','centimeters','position',[2 1 25 20]);
set(ha,'ZDir', 'Reverse','fontsize',10);
%%
loca=load('../test_xyz.txt');
figure
plot(loca(:,1),loca(:,2),'k*')
ylabel('X ','fontsize',15);xlabel('Y','fontsize',15);
set(gca,'fontsize',15);



