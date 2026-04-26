N_obj= 700;
O=phantom3d(N_obj);

Ne = 1;

O = O +1;
Zr= O(:,:,N_obj/2); % change here

channel= [240];
O=O(:,:,channel);

Zi = sum(O,3); % switch Zi and Zr position here
object_true=Zr+1i*Zi;

N_scans_x = 12; % number of scan we have is n_scan_x^2
dd=1;
N = (10/dd)^2;   
stepSize_true = 5e-7; %scan step size
Nc_avg = 1000; % determine the noise; if inf, then no noise if want noise 10^6

sim_pty_step_scan;

Zr = Zr(min(ind_y_lb):max(ind_y_ub),min(ind_x_lb):max(ind_x_ub),:); % Now Zr is the real part
O = imag(obj); % O is the imaginery Part


figure, 
set(gcf, 'Position',  [100, 100, 500, 200])
subplot(1,2,1);imagesc(Zr);colorbar; title('current real (object) saved'); axis square
subplot(1,2,2);imagesc(O);colorbar; title('current imaginary (object) saved'); axis square


p = probe_true;
Np=size(p,1);
d=[];

Nc_avg_fluor = 4e5; % 1e7 -> 0.5% noise 3e6 -> 1% noise 3e5 -> 3% noise 3e4 -> 10% noise

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

%% visualization

figure;
set(gcf, 'Position', [100, 100, 500, 200]);

subplot(1,2,1);
imagesc(d(:,:,1));
colorbar;
title('Noisy Fluor');
axis square;

subplot(1,2,2);
imagesc(d_true(:,:,1));
colorbar;
title('Noiseless Fluor');
axis square;



w_show = squeeze(w_true(:,:,1));

figure
set(gcf, 'Position', [100, 100, 500, 200])
imagesc(w_show); colorbar;
title('w true');
axis square; axis tight;



figure;
set(gcf, 'Position', [100, 100, 450, 350]);

imagesc(abs(p));
axis image; axis tight;
colorbar;
title('Probe Magnitude','FontSize',18,'FontWeight','bold');
%% saving



if ~exist('data_Eric/Np_'+string(Np),"dir")
    mkdir('data_Eric','Np_'+string(Np))
    addpath(genpath('data_Eric'))
end


save(['data_Eric/Np',num2str(Np),'_multimodal_',num2str(Ne),'_',num2str(stepSize_true),'_',num2str(N_scan),'_N_obj',num2str(N_obj),'_noise',num2str(Nc_avg),'_aperiodic.mat'],...
     'd','O','Zr','dp','ppX','ppY','dx','probe_true','obj','-v7.3');


%%
