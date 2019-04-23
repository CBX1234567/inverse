function [S_ff,T_e0]=RLOAD(B,nel,gcoord,emodule,poison,nodes,S_ff,sdof,T0,T_e0)
%STEPVP Summary of this function goes here
%   Detailed explanation goes here

    ELOAD=zeros(sdof,1);


   for  iel=1:nel
                
        matmtrx=zeros(3,3);
  
 	    [matmtrx]=fematiso(2,emodule,poison);
        
        
        nd(1)=nodes(iel,1);         %1st connected node for (iel)-th element
        nd(2)=nodes(iel,2);         %2nd connected node for (iel)-th element
        nd(3)=nodes(iel,3);         %3rd connected node for (iel)-th element
    
        x1=gcoord(nd(1),1); y1=gcoord(nd(1),2);     %coord values of 1st node
        x2=gcoord(nd(2),1); y2=gcoord(nd(2),2);     %coord values of 2nd node
        x3=gcoord(nd(3),1); y3=gcoord(nd(3),2);     %coord values of 3rd node
        
        t=(T0(nd(1))+T0(nd(2))+T0(nd(3)))/3;
        e0=9*10^(-6)*[1;1;0]*(t)*1.3;
      	area=0.5*(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2); %area of triangula
       
        for i=1:3
           T_e0(i,iel)=e0(i,1);
       end

      S_ff=cal_ELOAD(B{iel},nodes,S_ff,iel,area,matmtrx,e0);
    
    end

end



