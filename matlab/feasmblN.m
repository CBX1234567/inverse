
function [N]=feasmblN(N,Ne,index)

for i=1:3
    ii=index(i);
    for j=1:3
        jj=index(j);
        N(ii,jj)=N(ii,jj)+Ne(i,j);
    end
end

end

