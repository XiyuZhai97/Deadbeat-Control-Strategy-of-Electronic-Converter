warning off
%模型参数设置
clear

%load_L=9.5e-3;
Filter_C=220e-6;%滤波器电容
Sample_time=1e-6;
N=50;%滑动平均滤波窗口大小
%=========================
errorlistl=zeros(20,1);
errorlistr=zeros(20,1);
for loadiii=1:20
load_L=loadiii*1e-3;
load_R=5;
sim('danxiang_close_singlevol_zuni',0.04)
u =ScopeData1.signals(1).values;  %电容电压
iL=ScopeData1.signals(3).values; %电感电流
i2=ScopeData1.signals(2).values; %负载电流示波器值
i_load=zeros(length(iL),1);  %通过电感电流计算负载电流
for n=1:length(i_load)-1
    i_load(n,1)=iL(n)-Filter_C*(u(n+1)-u(n))/Sample_time;
end
i_load_filter0=moving_average_filter(i_load,N);%滑动平均值滤波，对i_load(通过电感电流计算得到的负载电流）进行滤波
i_load_filter=kalman_filter(i_load_filter0,1e-6,1e-3,0,3); %再进行一次卡尔曼滤波
%draw_filter(iL,iL_filter,i_load,i_load_filter,i2,u)
%超定方程组的最下二乘解   转化为  法方程组的解问题 Ax=b,A:n行m列，

 for n=1:length(i_load_filter)-1000
    A(n,1)=(i_load_filter(n+1)-i_load_filter(n))/Sample_time;
    A(n,2)=i_load_filter(n);
    b(n,1)=u(n);
    Z=A\b; %左除法求解
%     error1_L(n,1)=100*(load_L-Z(1))/load_L;
%     error1_R(n,1)=100*(load_R-Z(2))/load_R;
 end

% figure(1)
% plot(error1_L(500:length(error1_L),:))
% hold on
% plot(error1_R(500:length(error1_R),:))
% legend('电感误差','电阻误差')
% fprintf("实际L=%.5f,R=%.5f \n算得L=%.5f,R=%.5f\n",load_L,load_R,Z(1),Z(2))
% fprintf("L误差=%.2f%%,R误差=%.2f%% \n",100*(load_L-Z(1))/load_L,100*(load_R-Z(2))/load_R)
errorlistl(loadiii)=100*(load_L-Z(1))/load_L;
errorlistr(loadiii)=100*(load_R-Z(2))/load_R;
end
figure(1)
plot(errorlistl)
hold on
plot(errorlistr)
ytickformat('%g %%')
xtickformat('%g mH')
% title('单相闭环负载参数辨识 R=5\Omega')
legend('电感误差%','电阻误差%')
xlabel('负载电感') 
ylabel('误差百分比') 

% %直接用负载电流值算==============================================================================
%  for n=1:length(i2)-1000
%     A(n,1)=(i2(n+1)-i2(n))/Sample_time;
%     A(n,2)=i2(n);
%     b(n,1)=u(n);
%     Z=A\b; %左除法求解
%     error1_L(n,1)=100*(load_L-Z(1))/load_L;
%     error1_R(n,1)=100*(load_R-Z(2))/load_R;
%  end
% 
% figure(2)
% plot(error1_L(500:length(error1_L),:))
% hold on
% plot(error1_R(500:length(error1_R),:))
% legend('电感误差','电阻误差')
% fprintf("实际L=%.5f,R=%.5f \n算得L=%.5f,R=%.5f\n",load_L,load_R,Z(1),Z(2))
% fprintf("L误差=%.2f%%,R误差=%.2f%% \n",100*(load_L-Z(1))/load_L,100*(load_R-Z(2))/load_R)
