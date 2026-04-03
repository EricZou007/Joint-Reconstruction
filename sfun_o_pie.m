function [f,g,H,gtot] = sfun_o_pie(x,probe, dp, ind_b, N_obj, Np, indices)

global p_overlap
obj = reshape(realToComplex(x),N_obj,N_obj);

f=0;
gz=zeros(N_obj,N_obj);
if(nargout>=3)
    H=sparse(N_obj^2*2,N_obj^2*2);
    % dH=sparse(N_obj^2*2,1);
    % Hu=H;
    F=fftshift(fft(ifftshift(eye(Np))));
    F=kron(F,F);
    if(nargout==4)
        gtot=sparse(length(indices),N_obj^2*2);
    end
end
k=1;
indices=reshape(indices,1,length(indices));
for i=indices; 
    obj_roi = obj(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));
    psi =  obj_roi .* probe;

 
    %Fourier projection
    psi_old = psi;
    psi = fftshift(fft2(ifftshift(psi)));

    cmp2=abs(psi)+0e-6;
    %%====================data misfit
    res=cmp2-dp(:,:,i);
    % p_overlap_roi=p_overlap(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2));
    % res=res./p_overlap_roi;
    f = f + 0.5*norm(res(:)/(Np^1),2)^2;
    %%==========================
    psi2=psi.^2;
    psi21=conj(psi).^2;
    %%======feasibility
    psi = psi./cmp2.*dp(:,:,i);
    psi = fftshift(ifft2(ifftshift(psi)));
    %%=====================object misfit
    % res=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(psi_old)))./cmp2.*res./p_overlap_roi)));
    res=fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(psi_old)))./cmp2.*res)));

    sub_g = conj(probe).*res;
    %%==================
    gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))= gz(ind_b(i,3):ind_b(i,4),ind_b(i,1):ind_b(i,2))+sub_g;
    if(nargout>=3)
        [ind_X,ind_Y]=meshgrid(ind_b(i,1):ind_b(i,2),ind_b(i,3):ind_b(i,4));
        vecInd=sub2ind([N_obj,N_obj],ind_Y(:),ind_X(:));
        P=zeros(Np^2,N_obj^2);
        for k1=1:Np^2
            P(k1,vecInd(k1))=probe(k1);
        end
        cmp1=P'*P;
        dpv=dp(:,:,i);dpv=dpv(:);
        cmp23=cmp2.^3;


        cmp3=P'*F'*diag(dpv.*psi2(:)./cmp23(:))*conj(F)*conj(P)./Np^2; %% use FFT's unitary property
        cmp4=P'*F'*diag(dpv./cmp2(:))*F*P/Np^2; %% use FFT's unitary property

        % dH(1:N_obj^2)=dH(1:N_obj^2)+real(diag(cmp1))-1/2*real(diag(cmp4))+1/2*real(diag(cmp3));
        % dH([1:N_obj^2]+N_obj^2)=dH([1:N_obj^2]+N_obj^2)+real(diag(cmp1))-1/2*real(diag(cmp4))-1/2*real(diag(cmp3));

        %%=====full Hessian
        H([1:N_obj^2],[1:N_obj^2])=H(1:N_obj^2,1:N_obj^2)+real(cmp1)-1/2*real(cmp4)+1/2*real(cmp3);
        H([1:N_obj^2]+N_obj^2,[1:N_obj^2]+N_obj^2)=H([1:N_obj^2]+N_obj^2,[1:N_obj^2]+N_obj^2)+real(cmp1)-1/2*real(cmp4)-1/2*real(cmp3);
        H([1:N_obj^2], [1:N_obj^2]+N_obj^2)=H(1:N_obj^2,[1:N_obj^2]+N_obj^2)+imag(cmp1)-1/2*imag(cmp4)-1/2*imag(cmp3);
        H([1:N_obj^2]+N_obj^2, [1:N_obj^2])=H([1:N_obj^2]+N_obj^2,1:N_obj^2)+imag(cmp1.')-1/2*imag(cmp4.')-1/2*imag(cmp3.');
        %%============unknown part
        %Hu([1:N_obj^2],[1:N_obj^2])=Hu(1:N_obj^2,1:N_obj^2)+1/2*real(cmp3)-1/2*real(cmp4);%
        %Hu([1:N_obj^2]+N_obj^2,[1:N_obj^2]+N_obj^2)=Hu([1:N_obj^2]+N_obj^2,[1:N_obj^2]+N_obj^2)-1/2*real(cmp3)-1/2*real(cmp4);%
        %Hu([1:N_obj^2], [1:N_obj^2]+N_obj^2)=Hu(1:N_obj^2,[1:N_obj^2]+N_obj^2)-1/2*imag(cmp4)-1/2*imag(cmp3);
        %Hu([1:N_obj^2]+N_obj^2, [1:N_obj^2])=Hu([1:N_obj^2]+N_obj^2,1:N_obj^2)-1/2*imag(cmp3.')-1/2*imag(cmp4.');
        if(nargout==4)
            gtot(k,[vecInd';vecInd'+N_obj^2]) = sub_g;%vecProbe; 
        end
        k=k+1;
    end
end
 % f=f/length(indices);
g=complexToReal(gz);%./length(indices);
if(nargout>=3)
    H=(H+H')/2/length(indices);
end