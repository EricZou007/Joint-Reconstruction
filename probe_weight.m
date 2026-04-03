function [Pw, c, P] = probe_weight(probe,indices,N_obj,ind_b,alpha)
global alpha_p0
if (nargin==4)
    alpha = alpha_p0;
end
probe_s=conj(probe(:)).*probe(:);
Np=size(probe,1);
N_scan = size(ind_b,1);
Pw=zeros(N_obj^2*2,1);
% peak=max(probe_s);
c=zeros(N_obj,N_obj); %% count number of times being touched for each object pixel
% P=zeros(Np^2,N_obj^2,N_scan);

%%===========general permutationa matrix
for i=indices; 
    [ind_X,ind_Y]=meshgrid(ind_b(i,1):ind_b(i,2),ind_b(i,3):ind_b(i,4));
    
    vecInd=sub2ind([N_obj,N_obj],ind_Y(:),ind_X(:));
    Pw(vecInd)=Pw(vecInd)+probe_s;
    Pw(N_obj^2+vecInd)=Pw(vecInd);
        % c(vecInd)=c(vecInd)+abs(probe_s); %%check actual illumination intensity
        % c(ind_b(i,1):ind_b(i,2),ind_b(i,3):ind_b(i,4))=c(ind_b(i,1):ind_b(i,2),ind_b(i,3):ind_b(i,4))+1; %%count number of being illuminated;:w
        c(vecInd)=c(vecInd)+1; %%count number of being illuminated;:w
        
        % c(N_obj^2+vecInd)=c(vecInd);
        % for k1=1:Np^2
        %     P(k1,vecInd(k1),i)=probe(k1);
        % end
end  

%%%===========================
P=Pw./length(indices);
P(isnan(P))=0;
c_temp=repmat(c(:),2,1);
Pw = Pw/length(indices); %% for average weighted preconditioner
% Pw = Pw./c_temp; %%for per-pixel weighted preconditioner 
Pw (isnan(Pw))=0;
%  alpha=1e-4;
Pw = (1-alpha)*Pw+alpha*max(abs(probe(:)).^2);%.*(Pw~=0);
% Pw = Pw./1+alpha*max(abs(probe(:)).^2);%.*(Pw~=0);
