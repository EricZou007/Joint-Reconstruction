
global N Np Ne N_obj
global H_init
global beta1 thres
global fctn_o
global fctn_f

%% Load the true data

N_probe = 100;  % 16, 32, 66, 1e-7
N_scan = 144;
N_obj_size =650;
stepSize = 5e-7;
Nc_avg =inf;
eig_val_need = 0;
true_value = 0;
stepSize_true = stepSize;
Ne = 1;

load(['data_Eric/Np', num2str(N_probe), '_multimodal_', num2str(Ne), '_', ...
     num2str(stepSize), '_', num2str(N_scan), '_N_obj', num2str(N_obj_size),'_noise',num2str(Nc_avg), '_aperiodic.mat']);


Zr_true = Zr;
O_true = O;
v_o_true =[Zr(:); O(:)];
v_obj_true = [Zr(:); O(:)];
p = probe_true;

%% Load the noise measurement reconstructed data


Nc_avg = 1000;



load(['data_Eric/Np', num2str(N_probe), '_multimodal_', num2str(Ne), '_', ...
     num2str(stepSize), '_', num2str(N_scan), '_N_obj', num2str(N_obj_size),'_noise',num2str(Nc_avg), '_aperiodic.mat']);


% Zr_true = Zr;
% O_true = O;
% v_o_true =[Zr(:); O(:)];
% v_obj_true = [Zr(:); O(:)];
% p = probe_true;



H_init = 'standard';
dia_pert = 2;
do_setup;
thres = 1e-10; 
N = N_obj^2 * (Ne + 1);
NF = [0 * N; 0 * N; 0 * N];

v_p = complexToReal(p);

Zr = ones(N_obj, N_obj, Ne);
O = ones(N_obj, N_obj);
indices = 1:N_scan;
  

v_o_joint = [Zr(:); O(:)];
v_o_ptyo = [Zr(:); O(:)];
v_o_alter = [Zr(:); O(:)];
v_o_fluor = [Zr(:); O(:)];


v_o = [Zr(:); O(:)];
v_o_init = [Zr(:); O(:)];
fctn_o = @(o) sfun_o(o, p, dp);
fctn_f = @(o) func_conv_o(o, p, d);
%% initialization for joint reconstruction:
f_tot_joint = [];
f_tot_joint_ptyc = [];
f_tot_joint_fluor = [];
f_tot_ptyo = [];
mse_joint = [];
gradient_joint = [];
mse_joint_imag = [];
mse_joint_real = [];
%% Joint reconstruction process
outermax = 1;


iter = 100;


iter_joint = iter;
maxiter = iter_joint;
eig_val_need = 0;
separate_loss = 1;
%%
fctn_o = @(o) sfun_o(o, p, dp);
fctn_f= @(o) func_conv_o(o,p,d);
[~, grad_ptyo] = sfun_o(v_o_init, p, dp);
[~, grad_fluor] = func_conv_o(v_o_init(N_obj^2+1:end), p ,d);
% beta1 = sqrt((norm(grad_ptyo)))/sqrt((norm(grad_fluor)));
beta1 = ((norm(grad_ptyo)))/((norm(grad_fluor))) * 0.01 ;
disp(beta1);

%%

global lambda_tv tv_type eps_tv mu_tv
lambda_tv = 0;    % tune
tv_type   = 'iso';   % 'iso' or 'aniso'
eps_tv    = 1e-10;    % TV denom threshold
mu_tv     = 1;       % same as mu unless you intentionally scale imag part


%%
for k = 1:outermax   
   disp(['============ Joint Reconstruction, Iteration: ', num2str(k)])
      
   fctn_j = @(o) sfun_joint(o, p, d, dp);       
   [v_o_joint, f, g] = tn(v_o_joint, fctn_j, v_o_true, eig_val_need, separate_loss);
   f_tot = f.cost();
   mse_joint = [mse_joint;f.mse_values];
   mse_joint_imag = [mse_joint_imag; f.mse_values_imag];
   gradient_joint = [gradient_joint; f.gnorm()'];
     
   Zr_joint = reshape(v_o_joint(1: Ne*N_obj^2), N_obj, N_obj); % change index here
   O_joint = reshape(v_o_joint(Ne*N_obj^2+1: end), N_obj, N_obj, Ne);
   v_o_joint = [Zr_joint(:); O_joint(:)];    
   f_tot_joint = [f_tot_joint; f_tot'];
   f_tot_joint_ptyc = [f_tot_joint_ptyc; f.cost_ptyc];
   f_tot_joint_fluor = [f_tot_joint_fluor; f.cost_fluor];
end
%% initialization for ptyochopragy reconstructif_tot_ptyo = [];
 mse_ptyo = [];
mse_ptyo_real = [];
mse_ptyo_imag = [];
gradient_ptyo = [];
iter_ptyo = iter;
maxiter = iter_ptyo;
outermax = 1;
separate_loss = 1;
for k = 1:outermax
   disp(['============ Outer iteration ', num2str(k), ' / ', num2str(outermax), ' ============']);
      
   fctn_o = @(o) sfun_o(o, probe_true, dp);     
   [v_o_ptyo, f, g] = tn(v_o_ptyo, fctn_o, v_o_true, eig_val_need, separate_loss);
   f_tot = f.cost(); 
   gradient_ptyo_tot = f.gnorm();
   mse_ptyo = [mse_ptyo; f.mse_values];
   mse_ptyo_real = [mse_ptyo_real; f.mse_values_real];
   mse_ptyo_imag = [mse_ptyo_imag; f.mse_values_imag];
   Zr_ptyo = reshape(v_o_ptyo(1:Ne*N_obj^2), N_obj, N_obj, Ne);
   O_ptyo = reshape(v_o_ptyo(Ne*N_obj^2+1:end), N_obj, N_obj);
   v_o_ptyo = [Zr_ptyo(:); O_ptyo(:)];
   obj_ptyo = Zr_ptyo(:) + 1i * O_ptyo(:); 
     
   f_tot_ptyo = [f_tot_ptyo; f_tot'];
   gradient_ptyo = [gradient_ptyo; gradient_ptyo_tot'];
  
end
%% Fluorescence Reconstruction

lambda_tv = lambda_tv / beta1;    


iter_fluor = iter;
outermax = 1;
separate_loss = 0;
f_tot_fluor = [];
mse_fluor = [];
gradient_flour = [];
for k = 1:outermax
   disp(['============ Deconvolving Fluorescence Images, Iteration: ', num2str(k)])
   maxiter = iter_fluor;
   fctn_conv = @(o) func_conv_o(o, p, d);
   [x_sub, f, g] = tn(v_o_fluor(N_obj^2 * Ne+1:end), fctn_conv, v_o_true(N_obj^2 * Ne+1:end), eig_val_need, separate_loss);
   f_tot = f.cost(); 
   gradient_flour_tot = f.gnorm();
   mse_fluor = [mse_fluor; f.mse_values];
   Zr_fluor = reshape(v_o_fluor(1: N_obj^2 * Ne), N_obj, N_obj);
   O_fluor = reshape(x_sub, N_obj, N_obj, Ne);
  
   v_o_fluor = [Zr_fluor(:); O_fluor(:)];
   f_tot_fluor = [f_tot_fluor; f_tot'];
   gradient_flour = [gradient_flour; gradient_flour_tot'];
end

%% plot
outermax_joint = 1;
figure('Units','pixels','Position',[100 100 1350 1050]);

tiled = tiledlayout(4,4,'TileSpacing','compact','Padding','compact');

% Make default font size larger for the figure
set(groot,'DefaultAxesFontSize',10);
set(groot,'DefaultColorbarFontSize',10);

% Prepare data (unchanged)
obj_true   = realToComplex(v_o_true);
obj_ptyo   = realToComplex(v_o_ptyo);
obj_fluor  = realToComplex(v_o_fluor);
obj_joint  = realToComplex(v_o_joint);
obj        = realToComplex(v_o_init);

Zr_data  = { Zr_true(:,:,1), Zr_ptyo(:,:,1), Zr_fluor(:,:,1), Zr_joint(:,:,1) };
O_data   = { O_true,          O_ptyo,         O_fluor(:,:,1),  O_joint(:,:,1)  };
amp_data = { abs(obj_true),   abs(obj_ptyo),  abs(obj),        abs(obj_joint)  };
phs_data = { angle(obj_true), angle(obj_ptyo), angle(obj), angle(obj_joint) };

rowNames = {'True','Ptychography','Fluorescence','Joint(ours)'};

% Column-wise shared limits
zr_min = min(Zr_true(:,:,1), [], 'all');    zr_max = max(Zr_true(:,:,1), [], 'all');
o_min  = min(O_true(:));                    o_max  = max(O_true(:));

all_amps = [amp_data{1}(:); amp_data{2}(:); amp_data{3}(:); amp_data{4}(:)];
amp_cmin = min(all_amps); amp_cmax = max(all_amps);
if amp_cmin>=amp_cmax, amp_cmin=amp_cmin-1; amp_cmax=amp_cmax+1; end

true_phase = angle(obj_true);
phs_cmin = min(true_phase(:)); phs_cmax = max(true_phase(:));

% --- Capture axes handles so we can compute positions later
axH = gobjects(4,4);

for row = 1:4
    % Column 1: Real
    axH(row,1) = nexttile((row-1)*4 + 1);
    imagesc(Zr_data{row}); axis image; set(axH(row,1),'XTick',[],'YTick',[]);
    caxis(axH(row,1),[zr_min zr_max]);
    if row==1, title('Reconstruction (real)','FontWeight','bold','FontSize',14); end
    text(-0.1, 0.5, rowNames{row}, ...
    'Units','normalized', ...   % relative to axis
    'FontWeight','bold', ...
    'FontSize',25, ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', ...
    'Rotation',90);

    % Column 2: Imaginary
    axH(row,2) = nexttile((row-1)*4 + 2);
    imagesc(O_data{row}); axis image; set(axH(row,2),'XTick',[],'YTick',[]);
    caxis(axH(row,2),[o_min o_max]);
    if row==1, title('Reconstruction (imaginary)','FontWeight','bold','FontSize',14); end

    % Column 3: Modulus
    axH(row,3) = nexttile((row-1)*4 + 3);
    imagesc(amp_data{row}); axis image; set(axH(row,3),'XTick',[],'YTick',[]);
    caxis(axH(row,3),[amp_cmin amp_cmax]);
    if row==1, title('Reconstruction (modulus)','FontWeight','bold','FontSize',14); end

    % Column 4: Phase
    axH(row,4) = nexttile((row-1)*4 + 4);
    imagesc(phs_data{row}); axis image; set(axH(row,4),'XTick',[],'YTick',[]);
    caxis(axH(row,4),[phs_cmin phs_cmax]);
    if row==1, title('Reconstruction (phase)','FontWeight','bold','FontSize',14); end
end

cbW  = 0.015;   % colorbar width (normalized figure units)
gap  = 0.010;   % horizontal gap between axes column and colorbar

for col = 1:4
    posTop    = get(axH(1,col), 'Position');
    posBottom = get(axH(4,col), 'Position');

    yBottom = posBottom(2);
    yTop    = posTop(2) + posTop(4);
    height  = yTop - yBottom;
    xRight  = posTop(1) + posTop(3) + gap;

    cb = colorbar(axH(1,col), 'Location','eastoutside');
    cb.Position = [xRight, yBottom, cbW, height];
    cb.FontSize = 14;  % explicitly set font size for colorbar ticks
end



%% loss graph

num_iterations_loss = (iter_joint + 1) * outermax_joint;
figure('Units','inches','Position',[1 1 5 5], ...
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);   

hJ = loglog(1:length(f_tot_joint), f_tot_joint, '-o', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP = loglog(1:length(f_tot_ptyo), f_tot_ptyo,'-d', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % force axes to be equal
ax = gca;
grid(ax,'off');         
ax.XMinorGrid  = 'on';  
ax.YMinorGrid  = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha        = 0.6;
ax.MinorGridAlpha   = 0.3;

% pad X so first tick is at 0.75 and last at 1.2×
ax.XLim = [0.75, num_iterations_loss * 1.2];

allL = [f_tot_joint_ptyc(:); f_tot_ptyo(:)];
ax.YLim = [min(allL) * 0.75, max(allL) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('Loss Value','FontSize',12);
title('Loss comparison','FontSize',14);
legend([hJ,hP], ...
       {'Joint Method','Ptychography Method'}, ...
       'Location','best','FontSize',12);

%% loss on ptyc scale


figure('Units','inches','Position',[1 1 5 5], ...           
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);   
hJ = loglog(1:length(f_tot_joint_ptyc), f_tot_joint_ptyc,'-o', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP = loglog(1:length(f_tot_ptyo), f_tot_ptyo,'-d', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square            
ax = gca;
grid(ax,'off');         
ax.XMinorGrid  = 'on';  
ax.YMinorGrid  = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha        = 0.6;
ax.MinorGridAlpha   = 0.3;

% pad X so first tick is at 0.75 and last at 1.2×
ax.XLim = [0.75, num_iterations_loss * 1.2];

allL = [f_tot_joint_ptyc(:); f_tot_ptyo(:)];
ax.YLim = [min(allL) * 0.75, max(allL) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('Loss Value','FontSize',12);
title('Loss comparison on Ptyc Scale','FontSize',14);
legend([hJ,hP], ...
       {'Joint Method','Ptychography Method'}, ...
       'Location','best','FontSize',12);

%% fluor loss plot

outermax_joint      = 1;
num_iterations_loss = (iter_joint + 1) * outermax_joint;


% ————————————————————————————————————————————————
% 1) LOSS PLOT
figure('Units','inches','Position',[1 0.96 5 4.8], ...            % make window square
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);    % ensure export is square
hJ = loglog(1:length(f_tot_joint_fluor), f_tot_joint_fluor,'-o', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP = loglog(1:length(f_tot_fluor), f_tot_fluor,'-d', ...
            'LineWidth',1.5,'MarkerSize',4,'Color',[0.5 0 0.5]); % Purple
hold off;

axis square             % force axes to be equal
ax = gca;
grid(ax,'off');         
ax.XMinorGrid  = 'on';  
ax.YMinorGrid  = 'on';
ax.MinorGridLineStyle = ':';
ax.GridAlpha        = 0.6;
ax.MinorGridAlpha   = 0.3;

% pad X so first tick is at 0.75 and last at 1.2×
ax.XLim = [0.75, num_iterations_loss * 1.2];

allL = [f_tot_joint_fluor(:); f_tot_fluor(:)];
ax.YLim = [min(allL) * 0.75, max(allL) * 1.2];

xlabel('Iteration','FontSize',12);

ax.YAxis.Exponent = 0;

% Build custom y-tick labels: only keep 10^14
yt = ax.YTick;                         % e.g., [1e14 2e14 ... 7e14]
ylabs = repmat({''}, size(yt));        % blank all labels
[~,idx] = min(abs(yt - 1e14));         % find tick closest to 1e14
ylabs{idx} = '10^{14}';                % show only "10^{14}"

ax.YTickLabel = ylabs;
ax.TickLabelInterpreter = 'tex';       % render the exponent

ylabel('Loss Value','FontSize',12);
title('Loss comparison on Fluor Scale','FontSize',14);
legend([hJ,hP], ...
       {'Joint Method','Fluorescence Method'}, ...
       'Location','best','FontSize',12);






%%

% ————————————————————————————————————————————————


num_iterations_error = iter_joint * outermax_joint;

% 2) ERROR (MSE) PLOT
figure('Units','inches','Position',[1 1 5 5], ...            % same square size
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);
hJ2 = loglog(1:length(mse_joint), mse_joint,'-o', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP2 = loglog(1:length(mse_ptyo), mse_ptyo,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % square axes again
ax2 = gca;
grid(ax2,'off');
ax2.XMinorGrid  = 'on';
ax2.YMinorGrid  = 'on';
ax2.MinorGridLineStyle = ':';
ax2.GridAlpha        = 0.6;
ax2.MinorGridAlpha   = 0.3;

ax2.XLim = [0.75, num_iterations_error * 1.2];
allE = [mse_joint(:); mse_ptyo(:)];
ax2.YLim = [min(allE) * 0.75, max(allE) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('MSE','FontSize',12);
title('Error comparison','FontSize',14);
legend([hJ2,hP2], ...
       {'Joint Method','Ptychography Method'}, ...
       'Location','best','FontSize',12);


%% imaginary MSE error plot


num_iterations_error = iter_joint * outermax_joint;

% 2) ERROR (MSE) PLOT
figure('Units','inches','Position',[1 1 5 5], ...            
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);
hJ2 = loglog(1:length(mse_joint_imag), mse_joint_imag,'-o', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP2 = loglog(1:length(mse_fluor), mse_fluor,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0.5 0 0.5]); % Purple
hP3 = loglog(1:length(mse_ptyo_imag), mse_ptyo_imag,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % square axes again
ax2 = gca;
grid(ax2,'off');
ax2.XMinorGrid  = 'on';
ax2.YMinorGrid  = 'on';
ax2.MinorGridLineStyle = ':';
ax2.GridAlpha        = 0.6;
ax2.MinorGridAlpha   = 0.3;

ax2.XLim = [0.75, num_iterations_error * 1.2];
allE = [mse_joint_imag(:); mse_fluor(:); mse_ptyo_imag(:)];
ax2.YLim = [min(allE) * 0.75, max(allE) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('MSE','FontSize',12);
title('Imaginary Error comparison','FontSize',14);
legend([hJ2,hP2, hP3], ...
       {'Joint(Imaginary Part)','Fluorescence', "Ptychography(Imaginery Part)"}, ...
       'Location','best','FontSize',12);

%% Real MSE error plot

num_iterations_error = iter_joint * outermax_joint;

% 2) ERROR (MSE) PLOT
figure('Units','inches','Position',[1 1 5 5], ...            
       'PaperUnits','inches','PaperPosition',[0 0 5 5]);
hJ2 = loglog(1:length(gradient_joint), gradient_joint,'-o', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0 0 1]); % Blue
hold on;
hP2 = loglog(1:length(gradient_flour), gradient_flour,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[0.5 0 0.5]); % Purple
hP3 = loglog(1:length(gradient_ptyo), gradient_ptyo,'-d', ...
             'LineWidth',1.5,'MarkerSize',4,'Color',[1 0.5 0]); % Orange
hold off;

axis square             % square axes again
ax2 = gca;
grid(ax2,'off');
ax2.XMinorGrid  = 'on';
ax2.YMinorGrid  = 'on';
ax2.MinorGridLineStyle = ':';
ax2.GridAlpha        = 0.6;
ax2.MinorGridAlpha   = 0.3;

ax2.XLim = [0.75, num_iterations_error * 1.2];
allE = [gradient_joint(:); gradient_flour(:); gradient_ptyo(:)];
ax2.YLim = [min(allE) * 0.75, max(allE) * 1.2];

xlabel('Iteration','FontSize',12);
ylabel('MSE','FontSize',12);
title('Gradient norm comparison','FontSize',14);
legend([hJ2,hP2, hP3], ...
       {'Joint','Fluorescence', "Ptychography"}, ...
       'Location','best','FontSize',12);