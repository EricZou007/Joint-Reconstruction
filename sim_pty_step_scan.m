addpath('utility_function')

%load('multimodal_1_1e-07_100_aperiodic.mat')

%% parameters
% N_scans_x = 6 ; % number of scan positions along horizontal direction
N_scans_y = N_scans_x; % number of scan positions along vertical direction
% dd=1;
% N = (4/dd)^2;   % size of diffraction pattern in pixel. only square dp allowed
% stepSize_true = 1e-7/dd/1; %scan step size
maxPosError = 1; %largest randrom position error
%Nc_avg = inf; %average photon count per detector pixel. For poisson noise, SNR = sqrt(Nc_avg);
% Nc_avg = 10^6;

Ls = N*1e-6/dd; %determine probe size % out of focus

% Ls = 0; % at focus

if(~exist('object_true','var'))
    %% load test object
    disp('Loading test object...')
    sample='cameraman';%'chip';% 
    if(strcmp(sample,'phantom'))
        load('phantom_50um_10nm.mat')
        projection=projection+1;
    elseif(strcmp(sample,'rice'))
        load('rice_50um_10nm.mat')
    elseif(strcmp(sample,'cameraman'))
        %load('cameraman_baboon512.mat')
        load('cameraman512.mat')
        sample = 'multimodal_1';
    elseif(strcmp(sample,'chip'));
        load('8.8kV_G7_50um_10nm.mat')
    elseif(strcmp(sample,'rice'));
        load('rice_50um_10nm.mat')
    end
    % projection=projection(1:512,1:512);
    projection(projection==0)=1;
    % projection_m = interp2(abs(projection),2);
    % projection_p = interp2(angle(projection),2);
    % whos projection_p
    % projection= projection_m.*exp(1i.*projection_p);
    O_magnitude_true = abs(projection)./max(abs(projection(:)));
    O_phase_true = angle(projection);
    O_phase_true = O_phase_true - min(O_phase_true(:));
    O_phase_true = O_phase_true ./ max(O_phase_true(:)) * 0.4; %rescale phase a little bit
    object_true = O_magnitude_true.*exp(1i*O_phase_true);
    % object_true = object_true(1:2000,1:2000); %crop the object. No need for such large FOV
end

N_obj = size(object_true,1); %only square object allowed
ind_obj_center = floor(N_obj/2)+1;

%% generate probe function
disp('Generating probe function...')
dx = 10e-9*dd; %physical pixel size
energy = 8.8;
lambda = 1.23984193e-9/energy;

probe_true =  generate_probe(N, lambda, dx, Ls, 'velo');

% FWHM_k = 0.2* stepSize_true;   
% probe_true = gaussian_probe(N, dx, FWHM_k, 'flat'); % gaussian probe

%%====================
overlapRatio = 1-stepSize_true/dx/N;

disp(['overlap ratio is ', num2str(overlapRatio)]);

%% generate scan positions
disp('Generating scan positions...')

pos_x = (1 + (0:N_scans_x-1) *stepSize_true);
pos_y = (1 + (0:N_scans_y-1) *stepSize_true);
% ==== centre this
pos_x  = pos_x - (mean(pos_x));
pos_y  = pos_y - (mean(pos_y));
[Y,X] = meshgrid(pos_x, pos_y);
ppX = X(:);
ppY = Y(:);
% avoid periodic artefacts, add some random offsets
ppX = ppX + 1e-10*(rand(size(ppX))*2-1);
ppY = ppY + 1e-10*(rand(size(ppY))*2-1);
% pos = pos +  relative_random_offset*randn(size(pos));

disp('Generating diffraction patterns...')

%calculate indicies for all scans
N_scan = length(ppX);
%position = pi(integer) + pf(fraction)
py_i = round(ppY/dx);
py_f = ppY - py_i*dx;
px_i = round(ppX/dx);
px_f = ppX - px_i*dx;
probe_true = shift(probe_true, dx, dx, px_f(1), py_f(1));

ind_x_lb = px_i - floor(N/2) + ind_obj_center;
ind_x_ub = px_i + ceil(N/2) -1 + ind_obj_center;
ind_y_lb = py_i - floor(N/2) + ind_obj_center;
ind_y_ub = py_i + ceil(N/2) -1 + ind_obj_center;
%%==========visualize three succesive steps of probe
cl={'r','g','b','k'};
app=abs(probe_true);
app=app(round(size(probe_true,1)/2),:);
%====visualize  overlap level in 1D
% figure, for i=1:length(cl), plot(linspace(ind_x_lb(i),ind_x_ub(i),length(app)),app,'.-','color',cl{i});hold on; end
%%%====visualize covered object region
% figure, imagesc(abs(object_true(min(ind_x_lb):max(ind_x_ub),min(ind_y_lb):max(ind_y_ub))));
% %%=======================================
dp = zeros(N,N,N_scan);
dp_true = zeros(N,N,N_scan);
obj=object_true(min(ind_y_lb):max(ind_y_ub),min(ind_x_lb):max(ind_x_ub));

% mag   = abs(obj);
% phase = angle(obj);
% 
% magN = mag ./ (max(mag(:)) + eps);   
% obj  = magN .* exp(1i*phase);



N_obj=size(obj,1);
snr = ones(N_scan,1)*inf; %signal-to-noise ratio of each diffraction pattern

% f = waitbar(0,'1','Name','Simulating diffraction patterns...',...
%      'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');
% setappdata(f,'canceling',0);
for i=1:N_scan
    % Check for clicked Cancel button
    % if getappdata(f,'canceling')
    %     break
    % end
    % Update waitbar and message
% waitbar(i/N_scan,f,sprintf('No.%d/%d',i,N_scan))
    
    probe_s = probe_true;
    % probe_s = shift(probe_true, dx, dx, px_f(i), py_f(i));
    obj_roi = object_true(ind_y_lb(i):ind_y_ub(i),ind_x_lb(i):ind_x_ub(i));
    psi =  obj_roi .* probe_s;
    
    %FFT to get diffraction pattern
    dp_true(:,:,i) = abs(fftshift(fft2(ifftshift(psi)))).^2;
    dp(:,:,i) = dp_true(:,:,i);
    
    %Add poisson noise
    if Nc_avg<inf
        disp('add poisson noise')

        
        dp_true_temp = dp_true(:,:,i);
        eta = sum(dp_true_temp(:))/(N^2*Nc_avg);

        dp_temp = dp_true_temp / eta;
        dp_temp = poissrnd(dp_temp);
        dp_temp = dp_temp*eta;
        dp(:,:,i) = dp_temp;

        noise_level_ptych(i) = sqrt(eta / mean(dp_true_temp, "all") ) * 100;
        
        fprintf('Frame %d: dp noise = %.3f%%\n', i, noise_level_ptych(i));
    end


end

% O = real(obj);
% Zi = imag(obj);
% 
% d = conv2(O,abs(probe_true),'same');

disp('done');
%save(['../data/',sample,'_',num2str(stepSize_true),'_',num2str(N_scan),'_aperiodic.mat'],'dp','ppX','ppY','dx','probe_true','obj','O','Zi','d','-v7.3');
%save(['multimodal_',num2str(Ne),'_',num2str(stepSize_true),'_',num2str(N_scan),'_aperiodic.mat'],'d','O','Zi','dp','ppX','ppY','dx','probe_true','obj','-v7.3');

figure, 
set(gcf, 'Position',  [100, 100, 500, 200])
subplot(1,2,1);imagesc(abs(obj));colorbar; title('abs(object)'); axis square
subplot(1,2,2);imagesc(angle(obj));colorbar; title('angle(object)'); axis square

% delete(f)

return
%% reconstruction
Niter = 50;
%advanced options
opt.iter_probe_update = 101; %iteration number for starting probe update
%opt.iter_pos_corr = 20; %iteration number for starting random position correction
%opt.N_probes = 3; %number of probe modes in mixed states pty

opt.obj0=rand(N_obj,N_obj)+1i*rand(N_obj,N_obj);

[obj_recon, probe_recon] = ePIE(dp, ppX, ppY, dx, probe_true, Niter);

figure
subplot(1,2,1)
imagesc(abs(obj_recon)); axis image; colormap gray
title('reconstructed object magnitude')

subplot(1,2,2)
imagesc(angle(obj_recon)); axis image; colormap gray
title('reconstructed object phase')

figure
imagesc(abs(probe_recon)); axis image; colormap jet
title('reconstructed probe magnitude')

