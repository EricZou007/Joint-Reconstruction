% A=imread('cameraman.tif');
% A = double(A);
% B=load('mandrill.mat');
% B = B.X;
% B = B(150:405, 150:405);
% Z=A+B*1i;
% a=max(max(abs(Z)));
% Z = Z./a;
% Zr = real(Z);
% Zi = imag(Z);
% Ne = 1;
% disp(size(Zi))
% object_true=Zr+1i*Zi;
% N_scans_x = 10;
% dd=1;
% N = (6/dd)^2; 
% stepSize_true = 2e-7; %scan step size
% Nc_avg = inf;
% sim_pty_step_scan;
% Zr = Zr(min(ind_y_lb):max(ind_y_ub),min(ind_x_lb):max(ind_x_ub),:); % Now Zr is the real part
% O = imag(obj); % O is the imaginery Part
% 
% 
% figure, 
% set(gcf, 'Position',  [100, 100, 500, 200])
% subplot(1,2,1);imagesc(Zr);colorbar; title('current real (object) saved'); axis square
% subplot(1,2,2);imagesc(O);colorbar; title('current imaginary (object) saved'); axis square
% 
% 
% p = probe_true;
% Np=size(p,1);
% d=[];
% 
% Nc_avg_fluor = inf;
% for i=1:Ne
%   d(:,:,i)=conv2(squeeze(O(:,:,i)), abs(p).^2, 'same');
%   d_true_temp = d(:,:,i);
%   d_true  = d_true_temp;
%   if Nc_avg_fluor < inf
%       disp('add poisson noise to d')
%       eta =  sum(dp_true(:))/(N^2*Nc_avg_fluor);
%       d_norm = d(:,:,i) / eta;
%       d_noisy = poissrnd(d_norm);
%       d(:,:,i) = d_noisy * eta;
%       noise_fluor_level = sqrt(eta / mean(d_true, "all") ) * 100;
%       fprintf('fluorescence noise = %.3f%%\n', noise_fluor_level);
%   end
% end
% if ~exist('data_Eric/Np_'+string(Np),"dir")
%   mkdir('data_Eric','Np_'+string(Np))
%   addpath(genpath('data_Eric'))
% end
% 
% save(['data_Eric/cameraman_Np',num2str(Np),'_multimodal_',num2str(Ne),'_',num2str(stepSize_true),'_',num2str(N_scan),'_N_obj',num2str(N_obj),'_noise',num2str(Nc_avg),'_aperiodic.mat'],...
%     'd','O','Zr','dp','ppX','ppY','dx','probe_true','obj','-v7.3');




%% 

A = im2double(imread('cameraman.tif'));   
mag = A - min(A(:));
mag = mag ./ (max(mag(:)) + eps);                                 

S = load('mandrill.mat');                
B = im2double(S.X);
B = B(150:405, 150:405);                 

% Map mandrill intensities -> phase in [-pi, pi]
Bmin = min(B(:));
Bmax = max(B(:));
phase = (B - Bmin) / (Bmax - Bmin);    
phase = 2*pi*phase - pi;                

% 3) complex object
object_true = mag .* exp(1i * phase);

Zr = real(object_true);
% Zi = angle(object_true);

%% 

Ne = 1;


N_scans_x = 6; 
dd=1;
N = (6/dd)^2;  
stepSize_true = 2e-7; %scan step size
Nc_avg = 1000;
sim_pty_step_scan;

Zr = Zr(min(ind_y_lb):max(ind_y_ub),min(ind_x_lb):max(ind_x_ub),:); % Now Zr is the real part

O = imag(obj);

figure, 
set(gcf, 'Position',  [100, 100, 500, 200])
subplot(1,2,1);imagesc(Zr);colorbar; title('current real (object) saved'); axis square
subplot(1,2,2);imagesc(O);colorbar; title('current imaginary (object) saved'); axis square

p = probe_true;
Np=size(p,1);

d=[];
Nc_avg_fluor = 1e5; % 1e7 -> 0.5% noise 3e6 -> 1% noise 3e5 -> 3% noise 3e4 -> 10% noise

w_true = (( max(O, 0) ));

for i=1:Ne
    d(:,:,i)=conv2(squeeze(w_true(:,:,i)), abs(p).^2, 'same'); 
    d_true_temp = d(:,:,i);
    d_true  = d_true_temp;

    if Nc_avg_fluor < inf
       disp('add poisson noise to d')

       eta =  sum(dp_true(:))/(N^2*Nc_avg_fluor);

       d_norm = d(:,:,i) / eta;
       d_noisy = poissrnd(d_norm);
       d(:,:,i) = d_noisy * eta;

       noise_fluor_level = sqrt(eta / mean(d_true, "all") ) * 100;

       fprintf('fluorescence noise = %.3f%%\n', noise_fluor_level);
    end
end

if ~exist('data_Eric/Np_'+string(Np),"dir")
   mkdir('data_Eric','Np_'+string(Np))
   addpath(genpath('data_Eric'))
end

save(['data_Eric/cameraman_Np',num2str(Np),'_multimodal_',num2str(Ne),'_',num2str(stepSize_true),'_',num2str(N_scan),'_N_obj',num2str(N_obj),'_noise',num2str(Nc_avg),'_aperiodic.mat'],...
     'd','O','Zr','dp','ppX','ppY','dx','probe_true','obj','w_true','-v7.3');

