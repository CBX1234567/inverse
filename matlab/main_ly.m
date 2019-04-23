
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%
% �ȵ���(��һ��߽�����+�ڶ���߽�����)

%% �����������T0���ڵ��¶ȣ���newnodestress���ڵ�Ӧ������tdisp���ڵ�λ�ƣ�




%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
format long             %��Ч����16λ
fid_inp=fopen('lyu.dat','r');
linestr=write_echo_file(fid_inp);                   
nnode=str2num(linestr);                  % ���ַ��������
linestr=write_echo_file(fid_inp);
nel=str2num(linestr);

for inode=1:nnode
    clear tmp
    linestr=write_echo_file(fid_inp);
    tmp= str2num(linestr);
    num=tmp(1);
    indexnode(inode)=num;
    gcoord(num,1)=tmp(2);
    gcoord(num,2)=tmp(3);  
end

for iel=1:nel
    clear tmp
    linestr=write_echo_file(fid_inp);
    tmp= str2num(linestr);
    num=tmp(1);
    indexele(inode)=num;%ÿ����Ԫ����Щ�ڵ㹹��
    nodes(num,1)=tmp(2);
    nodes(num,2)=tmp(3);
    nodes(num,3)=tmp(4);
end   
nnel=3;
ndof=1;
sdof=nnode*ndof;        %total system dofs����ϵͳ�����ɶ�
edof=nnel*ndof;         %degrees of freedom per elementÿ����Ԫ�����ɶ�

% for i=1:nnode
%     x(i)=gcoord(i,1);
%     y(i)=gcoord(i,2);
% end  


%--------------------------------------------
% boundary condition pionts
%--------------------------------------------
     fm=0;
     fk=[];
     bcT=[];
     bcq=[];
     for i=1:nnode%nnode�����
         t=gcoord(i,2);%��������yֵ
         if t>0.02399999
             fm=fm+1;
             bcT(fm)=i;%s��y����0.0239999�Ľ����ѡ����
         end
     end     
     
     fm=0;
     fk=[];
   	 for i=1:nel%nelΪ��Ԫ��
          kum=0;
             for i1=1:3
	         n=nodes(i,i1); %ȡ��i����Ԫ��i1�еĽڵ���
	         t2=gcoord(n,2);%��������Yֵ
                if (t2==0.0)
	             kum=kum+1;
	             fk(kum,1)=n;
                end 
             end
             if (kum==2) %�����i����Ԫ�����������yֵ����Ϊ0
	         fm=fm+1;
	         bcq(1,fm)=fk(1,1);
	         bcq(2,fm)=fk(2,1);         
             end 
     end
     fm=0;
     fk=[];  
%   for h=0:6
%      for i=1:nel
%           kum=0;
% 	     for i1=1:3
%              n=nodes(i,i1); 
% 	         t1=gcoord(n,1);
%              t2=gcoord(n,2);
%              t3=(t1-h*0.0085);
%              if (t2>0.00399999&&0.0029999<t3&&t3<=0.003001)
%                   kum=kum+1;
% 	              fk(kum,1)=n;
%              end
%              if (kum==2) 
% 	             fm=fm+1;
% 	             bch(1,fm)=fk(1,1);
% 	             bch(2,fm)=fk(2,1);   
%              end
%          end
%      end
%   end
%     
%     for h=0:6
%      for i=1:nel
%           kum=0;
% 	     for i1=1:3
%              n=nodes(i,i1); 
% 	         t1=gcoord(n,1);
%              t2=gcoord(n,2);
%              t3=(t1-h*0.0085);
%              if (t2>0.00399999&&0.0059999<t3&&t3<=0.006001)
%                   kum=kum+1;
% 	              fk(kum,1)=n;
%              end
%              if (kum==2) 
% 	             fm=fm+1;
% 	             bch(1,fm)=fk(1,1);
% 	             bch(2,fm)=fk(2,1);   
%              end
%          end
%      end
%     end

     for i=1:nel
          kum=0;
	     for i1=1:3
             n=nodes(i,i1);%ȡ��i����Ԫ��i1�еĽڵ��� 
	         t1=gcoord(n,1)*1000;%��i����Ԫ��i1�еĽڵ��x
             t2=gcoord(n,2)*1000;%��i����Ԫ��i1�еĽڵ��y
             if (t2>3.99999 && t1<3.0000001)
                  kum=kum+1;
	              fk(kum,1)=n;
             end
             if (kum==2) %�����i����Ԫ������������������
	             fm=fm+1;
	             bch(1,fm)=fk(1,1);
	             bch(2,fm)=fk(2,1);   
             end
         end
     end
       fm=0;
     fk=[];  
          for i=1:nel
          kum=0;
	     for i1=1:3
             n=nodes(i,i1); 
	         t1=gcoord(n,1)*1000;
             t2=gcoord(n,2)*1000;
             if (t2>3.99999 && t1>56.99999)
                  kum=kum+1;
	              fk(kum,1)=n;
             end
             if (kum==2) 
	             fm=fm+1;
	             bch1(1,fm)=fk(1,1);
	             bch1(2,fm)=fk(2,1);   
             end
         end
     end
   
kk=zeros(sdof,sdof);
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% ����k

for iel=1:nel
    
    nd(1)=nodes(iel,1);         %1st connected node for (iel)-th element
    nd(2)=nodes(iel,2);         %2nd connected node for (iel)-th element
    nd(3)=nodes(iel,3);         %3rd connected node for (iel)-th element
    
    x1=gcoord(nd(1),1); y1=gcoord(nd(1),2);     %coord values of 1st node
    x2=gcoord(nd(2),1); y2=gcoord(nd(2),2);     %coord values of 2nd node
    x3=gcoord(nd(3),1); y3=gcoord(nd(3),2);     %coord values of 3rd node
%     a1=x2*y3-x3*y2;            ���κ����õ�
%     b1=y2-y3;
%     c1=x3-x2;
%     
%     a2=x3*y1-x1*y3;
%     b2=y3-y1;
%     c2=x1-x3;
%     
%     a3=x1*y2-x2*y1;
%     b3=y1-y2;
%     c3=x2-x1;

    index=feeldof(nd,nnel,ndof);        %extract system dofs for the element

    area=0.5*(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2); %area of triangula
	area2=area*2;
    
    dhdx=(1/area2)*[(y2-y3) (y3-y1) (y1-y2)];       %derivatives w.r.t x
    dhdy=(1/area2)*[(x3-x2) (x1-x3) (x2-x1)];       %derivatives w.r.t y

    D=[30,0;0,30];
    
    B=fekine2d(nnel,dhdx,dhdy);               %kinematic matrice   B����
   
   	k1=B'*D*B*area;                 %element stiffness matrice ��Ԫ�նȾ���
   
   	kk=feasmbl1(kk,k1,index);                        %assemble element matrics
  
    
end
lfc1=length(bcT);              
lfc2=length(bcq);              
%  %%%%%%%%%%%% ʩ�ӱ߽�����
%   ff=zeros(sdof,1);
for q=10:10:20000
    
    ff=zeros(sdof,1);
    kkc = zeros(sdof,sdof);
        for i=1:lfc2
            m=bcq(1,i);
            n=bcq(2,i);
            si=sqrt((gcoord(n,1)-gcoord(m,1))^2+(gcoord(n,2)-gcoord(m,2))^2);%�൱�ڵ�Ԫ�ı߳�
       
            si=si*(q);
            ff(m)=ff(m)+si/2;
            ff(n)=ff(n)+si/2;
       
        end
      
        lfc3=length(bch);
        for i=1:lfc3
           m=bch(1,i);
           n=bch(2,i);
           si=sqrt((gcoord(n,1)-gcoord(m,1))^2+(gcoord(n,2)-gcoord(m,2))^2);
           si=si*(15);%15Ϊh
           kkc(m,n)=kkc(m,n)+si/6;
           kkc(n,m)=kkc(n,m)+si/6;
           kkc(m,m)=kkc(m,m)+si/3;
           kkc(n,n)=kkc(n,n)+si/3;
           ff(m)=ff(m)+25*(si/2);%25�¶�ֵ
           ff(n)=ff(n)+25*(si/2);
        end
        lfc4=length(bch1);
       for i=1:lfc4
        m=bch1(1,i);
        n=bch1(2,i);
        si=sqrt((gcoord(n,1)-gcoord(m,1))^2+(gcoord(n,2)-gcoord(m,2))^2);
        si=si*(15);
        kkc(m,n)=kkc(m,n)+si/6;
        kkc(n,m)=kkc(n,m)+si/6;
        kkc(m,m)=kkc(m,m)+si/3;
        kkc(n,n)=kkc(n,n)+si/3;
        ff(m)=ff(m)+100*(si/2);
        ff(n)=ff(n)+100*(si/2);
       end
       kkt = kk + kkc;
       for i=1:lfc1
           m=bcT(i);%��һ��
           si=10^10;
           kkt(m,m)=kkt(m,m)*si;
           ff(m)=kkt(m,m)*20;
       end
   %%%%%%%%%%%%%%%���������״��������
% N=zeros(sdof,sdof);
%for iel=1:nel 
%     nd(1)=nodes(iel,1);         %1st connected node for (iel)-th element
 %    nd(2)=nodes(iel,2);         %2nd connected node for (iel)-th element
  %   nd(3)=nodes(iel,3);         %3rd connected node for (iel)-th element
    
  %   x1=gcoord(nd(1),1); y1=gcoord(nd(1),2);     %coord values of 1st node
  %   x2=gcoord(nd(2),1); y2=gcoord(nd(2),2);     %coord values of 2nd node
  %   x3=gcoord(nd(3),1); y3=gcoord(nd(3),2);     %coord values of 3rd node

  %  index=feeldof(nd,nnel,ndof);        %extract system dofs for the element

  %   area=0.5*(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2); %area of triangula

        
     %    for m=1:3
       %      for n=1:3
          %       if m==n
          %           Ne(m,n)=1/6*area*2730*893;         % �κ������ֵı任
           %     else
              %       Ne(m,n)=1/12*area*2730*893;
             %    end
          %   end
      %   end
     %    index=feeldofN(nodes,3,ndof,iel);
        
      %   N=feasmblN(N,Ne,index);        
        
 % end
%T0=15*ones(nnode,1);
%     dt=0.5;
%     for tttt=1:500
     % toler=0;
     % T1=T0;
     % N1=N/dt;
      %M=N1+kk;
      %Z=(N1)*T0;
      %L=ff+Z;
      T0=kkt\(ff);
      %for i=1:sdof
          %  total=(T0(i,1)-T1(i,1))^2;
        %    toler=toler+total;  
     %   end
      %   toler=toler^(0.5);
     %    if toler<=0.0001
      %      break
      %   end
   
%         Tmax=T0(1);
%         for i=2:nnode
%             if Tmax<T0(i)
%                 Tmax=T0(i);
%             else
%                 Tmax=Tmax;
%             end
%         end
%         Tm(tttt)=Tmax;
%            t(tttt)= dt*tttt;
%            
%            T_e(:,tttt)=T0;
%      end
[X,Y,Z]=griddata(gcoord(:,1),gcoord(:,2),T0,linspace(0,0.06,201)',linspace(0,0.024,201));
         for i=1:201
             for j=1:201
                 if((X(1,i)>0 && X(1,i)<0.003) && (Y(j,1)>0.004 && Y(j,1)<0.024))
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.057 && X(1,i)<0.06 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.006 && X(1,i)<0.0115 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.0145 && X(1,i)<0.02 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.023 && X(1,i)<0.0285 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.0315 && X(1,i)<0.037 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.04 && X(1,i)<0.0455 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                 else if(X(1,i)>0.0485 && X(1,i)<0.054 && Y(j,1)>0.004 && Y(j,1)<0.024)
                     Z(j,i)=NaN;
                     end
                     end
                     end
                     end
                     end
                     end                
             end
                 end
             end
         end    
         [c,h]=contourf(X,Y,Z,15);
         colormap(jet);
         pcolor(X,Y,Z);shading interp
          axis off
        
         path_in='C:\Users\cbx\Desktop\�½��ļ���\image\';     % path_inΪ����·����������Ҫ�޸�?
         saveas(gcf,[path_in,num2str(10\q)],'jpg');    % ����ͼƬ��������������       
         %close;
        % break;
end   
         %changes=[t,Tmax];
   %save datachange.txt changes -ascii; 
  % save datachange.txt changes; 
 % nodedata=fopen('D:\image1\num2str(tttt).plt','w');               
 %fprintf(nodedata,'TITLE="data"\n');
 
 %fprintf(nodedata,'VARIABLES=,"X", "Y", "S"\n');
% 
 %fprintf(nodedata,'ZONE T="flow-field" , N= %8d,E=%8d,ET=TRIANGLE, F=FEPOINT\n',nnode,nel); % ע�������˫����
% 
 %for i=1:nnode
   %  fprintf(nodedata,'%10.10f,%10.10f,%10.10f\n',gcoord(i,1),gcoord(i,2),T0(i));

 
% for i=1:nel
 %    for j=1:3
%         fprintf(nodedata,'%d       ',nodes(i,j));  
   %  end
   %  fprintf(nodedata,'\n');
% end
 %end
 
%        changes=[t',Tm'];
%   save datachange.txt changes -ascii; 
% nodedata=fopen('Temperature3.plt','w');               
% 
% 
% fprintf(nodedata,'TITLE="data"\n');
% 
% fprintf(nodedata,'VARIABLES=,"X", "Y", "T"\n');
% for ii=1:500
%     
% fprintf(nodedata,'ZONE T="%d  " , N= %8d,E=%8d,ET=TRIANGLE, F=FEPOINT\n',ii,nnode,nel); % ע�������˫����
% 
% for i=1:nnode
%     fprintf(nodedata,'%10.10f,%10.10f,%10.10f\n',gcoord(i,1),gcoord(i,2),T_e(i,ii));
% end  
% 
% for i=1:nel
%     for j=1:3
%         fprintf(nodedata,'%d       ',nodes(i,j));  
%     end
%     fprintf(nodedata,'\n');
% end
% end