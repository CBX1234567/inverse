xm=[0,0;0,0.1;0.1,0;0.1,0.1;0.2,0;0.2,0.1];
nodes=[3,2,1;2,3,4;3,5,4;4,5,6];
nnode=6;
nelem=4;
nnel=3;
ndof=1;
sdof=nnode*ndof;        %total system dofs
edof=nnel*ndof;         %degrees of freedom per element

kk=zeros(sdof,sdof);
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% º∆À„k
for iel=1:nelem
    
    nd(1)=nodes(iel,1);         %1st connected node for (iel)-th element
    nd(2)=nodes(iel,2);         %2nd connected node for (iel)-th element
    nd(3)=nodes(iel,3);         %3rd connected node for (iel)-th element
    
    x1=xm(nd(1),1); y1=xm(nd(1),2);     %coord values of 1st node
    x2=xm(nd(2),1); y2=xm(nd(2),2);     %coord values of 2nd node
    x3=xm(nd(3),1); y3=xm(nd(3),2);     %coord values of 3rd node

    index=feeldof(nd,nnel,ndof);        %extract system dofs for the element

    area=0.5*(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2); %area of triangula
	area2=area*2;
    
    dhdx=(1/area2)*[(y2-y3) (y3-y1) (y1-y2)];       %derivatives w.r.t x
    dhdy=(1/area2)*[(x3-x2) (x1-x3) (x2-x1)];       %derivatives w.r.t y

    D=[1,0;0,1;];
    
    B=fekine2d(nnel,dhdx,dhdy);               %kinematic matrice   Bæÿ’Û
   
   	k1=B'*D*B*area;                 %element stiffness matrice
   
   	kk=feasmbl1(kk,k1,index);                        %assemble element matrics
  
    
end

ff=zeros(sdof,1);

       m=5;
       n=6;
       si=0.1;
       si=si*(20);
       kk(m,n)=kk(m,n)+si/6;
       kk(n,m)=kk(n,m)+si/6;
       kk(m,m)=kk(m,m)+si/3;
       kk(n,n)=kk(n,n)+si/3;
       ff(m)=ff(m)+0*(si/2);
       ff(n)=ff(n)+0*(si/2);
 
       m=1;
       n=2;
       si=0.1;
       si=si*(20);
       kk(m,n)=kk(m,n)+si/6;
       kk(n,m)=kk(n,m)+si/6;
       kk(m,m)=kk(m,m)+si/3;
       kk(n,n)=kk(n,n)+si/3;
       ff(m)=ff(m)+100*(si/2);
       ff(n)=ff(n)+100*(si/2);

[LL UU]=lu(kk);
utemp=LL\ff;
T0=UU\utemp