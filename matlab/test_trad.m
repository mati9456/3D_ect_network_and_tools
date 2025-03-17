%clear all;
%close all;

addpath fwdp/ % Folder containing files required for model preparation and forward problem computation
addpath invp/ % Folder containing files required for image reconstuction
addpath visualization/ % Folder containing files required for visualization of the results
addpath torchFuncts/ % C matrix shortening function is needed

disp_rocnstr = true;
streams = true;

if ~exist('modelInvp', 'var')
    load("modelInvp2.mat");
end

if ~exist('model_max', 'var')
    model_max = load("max2.mat");
    modelInvp.min.S = model_max.max.S;
    modelInvp.obj = modelInvp.min;
    modelInvp.max = model_max.max;
end

fov_ix = findIndexInvp(modelInvp, 'fovcenter');

load("minC.mat")
load("maxC.mat")

modelInvp.max.C_real = maxC;
modelInvp.min.C_real = minC;
modelInvp.real_data = true;

% Specify the folder containing the .mat files
folderPath = '.\zbior_test_trad';

matFiles = dir(fullfile(folderPath, '*.mat'));

%sample_size = length(matFiles);
sample_size = 500;

% Initialize arrays to store metrics for each file
MSE_list = zeros(sample_size, 1);
PSNR_list = zeros(sample_size, 1);
SSIM_list = zeros(sample_size, 1);
Correlation_list = zeros(sample_size, 1);
time_list = zeros(sample_size, 1);

for k = 1:sample_size
    modelINVP2 = modelInvp;

    fullFilePath = fullfile(folderPath, matFiles(k).name);
    data = load(fullFilePath);

    data.eps_map = flip(data.eps_map, 3);
    data.eps_map(data.eps_map == 2) = 1;

    esp_map = data.eps_map;
    modelINVP2.obj.C_real = data.C;
     
    tic
    %ronstr = semiLM(modelINVP2, 200, 1e+1, 1e-2,0).eps_map; % invp, no. of iterations, alpha
    % ronstr = PINV(modelINVP2,1e-0).eps_map;
    ronstr = LBP(modelINVP2).eps_map;
    ronstr(ronstr>3)=3;
    % ronstr = Landweber(modelINVP2, 200, 1e+2).eps_map; % invp, no. of iterations, alpha
    % ronstr = semiLM(modelINVP2, 200, 1e-1, 1e-2,0).eps_map;
    time = toc;
    time_list(k) = time;

    %ronstr = imresize3(ronstr, [64 64 64], 'linear');
    ronstr(setdiff(1:end,fov_ix)) = 1;

    %drawInvpMap(modelNVP, 'Landweber', 'permittivity', 'mm', ixInvp);
    if disp_rocnstr == true & sample_size < 20
        mpr(ronstr);
        mpr(esp_map);
    end

    % Calculate MSE
    mse = mean((esp_map(fov_ix) - ronstr(fov_ix)).^2);
    MSE_list(k) = mse; 
    
    % Calculate PSNR
    psnr_value = 10 * log10(max(esp_map.^2, [], 'all') / mse);
    PSNR_list(k) = psnr_value;
    
    % Calculate SSIM
    ssim_value = ssim(single(ronstr), single(esp_map));
    SSIM_list(k) = ssim_value;
    
    % Calculate Correlation
    correlation = corr(ronstr(fov_ix), esp_map(fov_ix));
    Correlation_list(k) = correlation;
    
    % Display metrics for the current file
    if streams == true
        fprintf('File: %s\n', fullFilePath);
        fprintf('MSE: %.4f, PSNR: %.4f, SSIM: %.4f, Correlation: %.4f, time: %.4f\n', mse, psnr_value, ssim_value, correlation, time);
    end
    disp(k);
end

% Compute statistics for all metrics
meanMSE = mean(MSE_list);
stdMSE = std(MSE_list);
medianMSE = median(MSE_list);

meanPSNR = mean(PSNR_list);
stdPSNR = std(PSNR_list);
medianPSNR = median(PSNR_list);

meanSSIM = mean(SSIM_list);
stdSSIM = std(SSIM_list);
medianSSIM = median(SSIM_list);

meanCorrelation = mean(Correlation_list);
stdCorrelation = std(Correlation_list);
medianCorrelation = median(Correlation_list);

meanTime = mean(time_list);
stdTime= std(time_list);
medianTime = median(time_list);

% Display overall statistics
fprintf('\nOverall Statistics:\n');
fprintf('MSE - Mean: %.4f, Std: %.4f, Median: %.4f\n', meanMSE, stdMSE, medianMSE);
fprintf('PSNR - Mean: %.4f, Std: %.4f, Median: %.4f\n', meanPSNR, stdPSNR, medianPSNR);
fprintf('SSIM - Mean: %.4f, Std: %.4f, Median: %.4f\n', meanSSIM, stdSSIM, medianSSIM);
fprintf('Correlation - Mean: %.4f, Std: %.4f, Median: %.4f\n', meanCorrelation, stdCorrelation, medianCorrelation);
fprintf('Time - Mean: %.4f, Std: %.4f, Median: %.4f\n', meanTime, stdTime, medianTime);

result.mse = MSE_list;
result.psnr = PSNR_list;
result.ssim = SSIM_list;
result.corr = Correlation_list;
result.time = time_list;

t = [datetime('now')];
t = strrep(datestr(t), ':', '-');
fname = ['lbp_inf_noise_results_500 ' t '.mat'];
save(fname, '-struct', 'result');