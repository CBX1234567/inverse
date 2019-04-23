function [STRSG]=STEPVP(B,nel,gcoord,ASDIS,emodule,poison,nodes,STRSG,sdof,T_e0)
%STEPVP Summary of this function goes here
%   Detailed explanation goes here

    TWOPI=6.283185308;

   for  iel=1:nel
        matmtrx=zeros(3,3);
 	    [matmtrx]=fematiso(2,emodule,poison);
	    [STRSG]=stress(B{iel},ASDIS,STRSG,matmtrx,iel,nodes,T_e0);
   end

end



