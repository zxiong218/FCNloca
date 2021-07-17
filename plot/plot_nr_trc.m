function plot_nr_trc(t1,data,scale,par)
%input: data, scale
%input: t1,data, scale
%input: t1,data, scale,par

maxv=max(max(data));
offset=maxv;
sz=size(data);
t=0:sz(2);
color='k';
if nargin<2
    scale=1.0;
elseif nargin==3
    t=t1;
    scale=1.0;
elseif nargin==4
    t=t1;
    scale=1.0;
    color=par(1);
end
for i=1:sz(1)
%h=plot(t,data(:,i)/scale+(n(2)-i+1)*offset,'color','k','LineWidth',1.0);
data_max=max(data(i,:));
h=plot(t,data(i,:)/(scale*data_max)+(sz(1)-i+1)*offset,'color','k','LineWidth',1.0);
hold on
end
axis tight
end