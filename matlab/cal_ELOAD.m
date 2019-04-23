function  [S_ff]=cal_ELOAD(bm,nodes,S_ff,iel,area,matmtrx,e0)


 for i=1:3
     nodLL(i)=nodes(iel,i);
end

eload=zeros(6,1);               
eload=bm'*matmtrx*e0*area; 

            %%%%%%%%%%%%%%%%%%%%%      xm  上面还有一个 xm
for i=1:3
    m=(i-1)*2+1;
    n=(i-1)*2+2;
    
    S_ff((nodLL(i)-1)*2+1,1)=S_ff((nodLL(i)-1)*2+1,1)+eload(m,1);
    S_ff((nodLL(i)-1)*2+2,1)=S_ff((nodLL(i)-1)*2+2,1)+eload(n,1);

end   
    


end


