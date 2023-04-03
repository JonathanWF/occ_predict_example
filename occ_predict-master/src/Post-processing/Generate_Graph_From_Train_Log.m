% Set up workspace
clc; 
format compact;
close all; 
clear;
warning('off');

%% Set up folders

% Project head
if ispc
    base_dir = fullfile('C:', 'projects', 'base_matlab');
    proj_dir = fullfile('C:', 'Users', 'jwfol', 'Desktop', 'Unet experiment Results', '300 epochs');
elseif isunix
    base_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'base_matlab');
    proj_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'proocc_quant_risk_score');
else
    fprintf(1, 'Unknown filesystem, please edit folder setup!\n');
    return;
end
versions = dir(proj_dir);
versions(1:2) = [];
str = "R";
versions = struct2cell(versions);
index = endsWith(versions(1,:),str);
versions = versions(1,index);

%Preallocate Train and Loss arrays
TrainAccuracy = zeros(300,4);
TrainLoss = zeros(300,4);
EvalAccuracy = zeros(300,4);
EvalLoss = zeros(300,4);
Min_Loss_Array = zeros(1,4);
Max_Acc_Array = zeros(1,4);
for k = 1:length(versions)
    log = versions{k};
    C = fopen(fullfile(proj_dir, log, 'train.log'));
    D = fscanf(C, "%c");
    E = strsplit(D, {'accuracy:', 'loss:'});

    size = (length(E)-1)/2;
    accuracy_array = [];
    loss_array = [];
    for i = 1:size
        string_number = 2*i;
        string_accuracy = string(E(1, string_number));
        string_accuracy = strtok(string_accuracy, [' ', ';']);
        string_accuracy = str2num(string_accuracy{1});
        accuracy_array = [accuracy_array, string_accuracy];
        string_number_2 = 2*i+1;
        string_loss = string(E(1, string_number_2));
        string_loss = extractBefore(string_loss, 7);
        string_loss = strtok(string_loss, ' ');
        string_loss = str2num(string_loss{1});
        loss_array = [loss_array, string_loss];
    end
    size_new = size/2;

    for j = 1:size_new
        train = 2*j-1;
        eval = 2*j;
        TrainAccuracy(j,k) = accuracy_array(train);
        TrainLoss(j,k) = loss_array(train);
        EvalAccuracy(j,k) = accuracy_array(eval);
        EvalLoss(j,k) = loss_array(eval);
    end
    Min_Loss_Array(:,k) = min(EvalLoss(:,k));
    Max_Acc_Array(:,k) = max(EvalAccuracy(:,k));
end
epoch_axis = 1:size_new;
a = figure;
plot(epoch_axis, TrainAccuracy(:,1), 'r', epoch_axis, TrainAccuracy(:,2), 'y',  epoch_axis, TrainAccuracy(:,3), 'g', epoch_axis, TrainAccuracy(:,4), 'b')
axis([0 300 0 1]);
legend('TrainAccuracy1', 'TrainAccuracy2A', 'TrainAccuracy3A',  'TrainAccuracy4A');
title('Train Accuracy: Active Learning')
ylabel('Accuracy')
xlabel('Epoch Number')

b = figure;
plot(epoch_axis, EvalAccuracy(:,1), 'r', epoch_axis, EvalAccuracy(:,2), 'y', epoch_axis, EvalAccuracy(:,3), 'g',  epoch_axis, EvalAccuracy(:,4), 'b')
axis([0 300 0 1]);
legend( 'EvalAccuracy1',  'EvalAccuracy2A', 'EvalAccuracy3A',  'EvalAccuracy4A');
title('Eval Accuracy: Active Learning')
ylabel('Accuracy')
xlabel('Epoch Number')

c = figure;
plot(epoch_axis, TrainLoss(:,1), 'r', epoch_axis, TrainLoss(:,2), 'y',  epoch_axis, TrainLoss(:,3), 'g', epoch_axis, TrainLoss(:,4), 'b')
axis([0 300 0 3.5]);
legend('TrainLoss1', 'TrainLoss2A', 'TrainLoss3A',  'TrainLoss4A');
title('Train Loss: Active Learning')
ylabel('Loss')
xlabel('Epoch Number')

d = figure;
plot(epoch_axis, EvalLoss(:,1), 'r', epoch_axis, EvalLoss(:,2), 'y', epoch_axis, EvalLoss(:,3), 'g',  epoch_axis, EvalLoss(:,4), 'b')
axis([0 300 0 3.5]);
legend( 'EvalLoss1',  'EvalLoss2A', 'EvalLoss3A',  'EvalLoss4A');
title('Eval Loss: Active Learning')
ylabel('Loss')
xlabel('Epoch Number')

