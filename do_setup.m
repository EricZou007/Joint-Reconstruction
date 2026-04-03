global nx_scan init_s lambda beta beta1
global N_scan N_obj p_overlap Np ind_b indices
more off;
%% load data and preprocessing



if(~exist('datastr','Var'))
    sample='chip';
    stepSize_true=1e-07;
    N_scan = 1600;
end
datastr=[sample,num2str(stepSize_true),'_',num2str(N_scan),'_aperiodic'];% 144'];% 

% load([datastr,'.mat'],'-mat');

nx_scan=sqrt(size(dp,3));
subind=[1:size(dp,3)];
rng('shuffle')
%%%%======= extract size of probe and object
Np = size(dp,1); %% size of the probe
N_obj=size(obj,1);
px = ppX(subind);
py = ppY(subind);
px = px(:);
py = py(:);
Ny_max = max(abs(round(min(py)/dx)-floor(Np/2)), abs(round(max(py)/dx)+ceil(Np/2)))*2+1;
Nx_max = max(abs(round(min(px)/dx)-floor(Np/2)), abs(round(max(px)/dx)+ceil(Np/2)))*2+1;
ind_obj_center = floor(N_obj/2)+1;
N_scan = length(px);
py_i = round(py/dx);
py_f = py - py_i*dx;
px_i = round(px/dx);
px_f = px - px_i*dx;

ind_x_lb = px_i - floor(Np/2) + ind_obj_center;
ind_x_ub = px_i + ceil(Np/2) -1 + ind_obj_center;
ind_y_lb = py_i - floor(Np/2) + ind_obj_center;
ind_y_ub = py_i + ceil(Np/2) -1 + ind_obj_center;
ind_b=[ind_x_lb,ind_x_ub,ind_y_lb,ind_y_ub];
if(~exist('probe_true'))
    probe_true=probe0;
end
probe_length=Np*1e-8;
overlap_ratio=(probe_length-stepSize_true)/Np/1e-8;
%%======== generate ground truth
obj_true=obj;
v_obj_true=complexToReal(obj_true(:));
v_p_true=complexToReal(probe_true(:));
v_true=[v_obj_true;v_p_true];
Nc_avg = inf;%%average photon count per detector pixel. For poisson noise, SNR = sqrt(Nc_avg);
if Nc_avg<inf
    disp('add poisson noise')
    for i=1:N_scan
        dp_true_temp = dp(:,:,i);
        dp_temp = dp_true_temp/sum(dp_true_temp(:))*(Np^2*Nc_avg);
        dp_temp = poissrnd(dp_temp);
        dp_temp = dp_temp*sum(dp_true_temp(:))/(Np^2*Nc_avg);
        snr(i) = mean((dp_true_temp(:)))/std(dp_true_temp(:) - dp_temp(:));
        dp(:,:,i) = dp_temp;
        % figure, subplot(1,2,1),imagesc(dp_temp);colorbar; subplot(1,2,2),imagesc(dp_true_temp);colorbar;
        % pause;
    end
end
snr=sqrt(Nc_avg);
dp_avg = sum(dp,3)/size(dp,3);
%%=======normalize initial probe

% load Au_probe_256
alpha_max = 21.4;      %aperture size(mrad)
%%=====generate initial probe using disk
% probe0 = setupAiryProbe(Np,15);%generateProbe(dx, Np, voltage, alpha_max ); % generate initial probe
energy = 8.8;
lambda = 1.23984193e-9/energy;
[probe0] =  generate_probe(Np, lambda, dx, Np*1e-6*dia_pert, 'velo');
[~,p_overlap]=probe_weight(probe0,1:N_scan,N_obj,ind_b,0.01); %%== Pw: generate weight as probe magnitude
invH_init=1;
dp = sqrt(dp);

%%=======initial guess
%%%%====== initialize probe and object
init_o='ones';%'rando';
if(strcmp(init_o,'rando'));
    v_obj0=rand(N_obj,N_obj)+1i*rand(N_obj,N_obj);
else
    v_obj0=1.*(ones(N_obj,N_obj)+1i*ones(N_obj,N_obj));
end
% v_obj0=v_obj_true+1e-2.*v_obj_true;%
% v_p0=v_p_true;%
v_obj0=complexToReal(v_obj0(:)./abs(v_obj0(:)));
v_p0=complexToReal(probe0(:));
v0=[v_obj0;v_p0];
