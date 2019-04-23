function [index] =feeldof(nodes,nnel,ndof,nel)
 


edof=nnel*ndof;
   k=0;
   for i=1:nnel
     start = (nodes(nel,i)-1)*ndof;
       for j=1:ndof
         k=k+1;
         index(k)=start+j;
       end
   end


