function [B]=fekine2d(nnel,dhdx,dhdy)

%------------------------------------------------------------------------
%  Purpose:
%     determine the kinematic equation between strains and displacements
%     for two-dimensional solids
%
%  Synopsis:
%     [kinmtx2]=fekine2d(nnel,dhdx,dhdy) 
%
%  Variable Description:
%     nnel - number of nodes per element
%     dhdx - derivatives of shape functions with respect to x   
%     dhdy - derivatives of shape functions with respect to y
%------------------------------------------------------------------------

 for i=1:nnel
 
 B(1,i)=dhdx(i);
 B(2,i)=dhdy(i);

 end
