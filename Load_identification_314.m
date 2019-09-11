warning off
%ģ�Ͳ�������
clear

%load_L=9.5e-3;
Filter_C=220e-6;%�˲�������
Sample_time=1e-6;
N=50;%����ƽ���˲����ڴ�С
%=========================
errorlistl=zeros(20,1);
errorlistr=zeros(20,1);
for loadiii=1:20
load_L=loadiii*1e-3;
load_R=5;
sim('danxiang_close_singlevol_zuni',0.04)
u =ScopeData1.signals(1).values;  %���ݵ�ѹ
iL=ScopeData1.signals(3).values; %��е���
i2=ScopeData1.signals(2).values; %���ص���ʾ����ֵ
i_load=zeros(length(iL),1);  %ͨ����е������㸺�ص���
for n=1:length(i_load)-1
    i_load(n,1)=iL(n)-Filter_C*(u(n+1)-u(n))/Sample_time;
end
i_load_filter0=moving_average_filter(i_load,N);%����ƽ��ֵ�˲�����i_load(ͨ����е�������õ��ĸ��ص����������˲�
i_load_filter=kalman_filter(i_load_filter0,1e-6,1e-3,0,3); %�ٽ���һ�ο������˲�
%draw_filter(iL,iL_filter,i_load,i_load_filter,i2,u)
%��������������¶��˽�   ת��Ϊ  ��������Ľ����� Ax=b,A:n��m�У�

 for n=1:length(i_load_filter)-1000
    A(n,1)=(i_load_filter(n+1)-i_load_filter(n))/Sample_time;
    A(n,2)=i_load_filter(n);
    b(n,1)=u(n);
    Z=A\b; %��������
%     error1_L(n,1)=100*(load_L-Z(1))/load_L;
%     error1_R(n,1)=100*(load_R-Z(2))/load_R;
 end

% figure(1)
% plot(error1_L(500:length(error1_L),:))
% hold on
% plot(error1_R(500:length(error1_R),:))
% legend('������','�������')
% fprintf("ʵ��L=%.5f,R=%.5f \n���L=%.5f,R=%.5f\n",load_L,load_R,Z(1),Z(2))
% fprintf("L���=%.2f%%,R���=%.2f%% \n",100*(load_L-Z(1))/load_L,100*(load_R-Z(2))/load_R)
errorlistl(loadiii)=100*(load_L-Z(1))/load_L;
errorlistr(loadiii)=100*(load_R-Z(2))/load_R;
end
figure(1)
plot(errorlistl)
hold on
plot(errorlistr)
ytickformat('%g %%')
xtickformat('%g mH')
% title('����ջ����ز�����ʶ R=5\Omega')
legend('������%','�������%')
xlabel('���ص��') 
ylabel('���ٷֱ�') 

% %ֱ���ø��ص���ֵ��==============================================================================
%  for n=1:length(i2)-1000
%     A(n,1)=(i2(n+1)-i2(n))/Sample_time;
%     A(n,2)=i2(n);
%     b(n,1)=u(n);
%     Z=A\b; %��������
%     error1_L(n,1)=100*(load_L-Z(1))/load_L;
%     error1_R(n,1)=100*(load_R-Z(2))/load_R;
%  end
% 
% figure(2)
% plot(error1_L(500:length(error1_L),:))
% hold on
% plot(error1_R(500:length(error1_R),:))
% legend('������','�������')
% fprintf("ʵ��L=%.5f,R=%.5f \n���L=%.5f,R=%.5f\n",load_L,load_R,Z(1),Z(2))
% fprintf("L���=%.2f%%,R���=%.2f%% \n",100*(load_L-Z(1))/load_L,100*(load_R-Z(2))/load_R)
