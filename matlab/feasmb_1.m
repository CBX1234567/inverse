function [ff]=feasmb_1(ff,f,index)
%----------------------------------------------------------
%  Purpose:
%     Assembly of element matrices into the system matrix
%
%  Synopsis:
%     [kk]=feasmbl1(kk,k,index)
%
%  Variable Description:
%     kk - system matrix
%     k  - element matri
%     index - d.o.f. vector associated with an element
%-----------------------------------------------------------

 
edof = length(index);
for i=1:edof
    ii=index(i);
    ff(ii)=ff(ii)+f(i);
end