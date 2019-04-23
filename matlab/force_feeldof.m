function [ index ] = force_feeldof(nd,nnel,ndof)
%FORCE_FEELDOF Summary of this function goes here
%   Detailed explanation goes here
%----------------------------------------------------------
%  Purpose:
%     Compute system dofs associated with each element 
%
%  Synopsis:
%     [index]=feeldof(nd,nnel,ndof)
%
%  Variable Description:
%     index - system dof vector associated with element "iel"
%     iel - element number whose system dofs are to be determined
%     nnel - number of nodes per element
%     ndof - number of dofs per node 
%-----------------------------------------------------------

   k=0;
   for i=1:nnel
       start = (nd(i)-1)*ndof;
       for j=1:ndof
         k=k+1;
         index(k)=start+j;
       end
   end

end

