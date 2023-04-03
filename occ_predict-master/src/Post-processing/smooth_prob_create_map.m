% Set up workspace
format compact;
close all;
clear;
clc;
warning('off');

%% Set up folders

% Project head
if ispc
    base_dir = fullfile('C:', 'projects', 'base_matlab');
    proj_dir = fullfile('C:', 'Users', 'jwfol', 'large_image', 'HistomicsTK', 'occ_predict_master');
    proj_2_dir = fullfile('C:', 'Users', 'jwfol', 'Desktop', 'TNT Inference Results', 'batch 4');
elseif isunix
    base_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'base_matlab');
    proj_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'occ_quant_risk_score');
else
    fprintf(1, 'Unknown filesystem, please edit folder setup!\n');
    return;
end

unsmoothed_source = fullfile(proj_2_dir);
smoothed_target = fullfile(proj_dir, 'data', 'smoothing', 'TNT smoothed');
unsmoothed_dir = dir(fullfile(unsmoothed_source));
unsmoothed_dir(1:2) = [];
unsmoothed_dir = natsortfiles({unsmoothed_dir.name});
label_target = fullfile(proj_dir, 'data', 'smoothing', 'TNT combined smoothed');

class_names = {'Ignore', 'Background', 'Lymphocytes', 'Stroma', 'Tumor',   'Mucosa', 'Adipose', 'Blood', 'Muscle Tissue', 'Nerves', 'Necrosis', 'Keratin Pearls', 'Junk'};
class_colors = {[255, 255, 255], [128,128,128], [255,255,0], [255, 0, 0], [0, 0, 255],   [0, 0, 255], [128, 128, 128], [0, 128,0], [128, 128, 0], [255, 128, 0], [0, 0, 0], [0, 0, 128], [255, 0, 255]};
num_classes = 13;

for idir = 1:floor(length(unsmoothed_dir)/(num_classes+2))
    init_z_stack = zeros(2000,2000,num_classes);
    binary_canvas = zeros(2000,2000,num_classes);
    index = (idir-1)*(num_classes+2)+1;
    file_names = unsmoothed_dir(index:index+num_classes-1);
    file_list = natsortfiles(file_names);
    original_index = unsmoothed_dir(index+num_classes+1);
    
    for i = 1:num_classes
        layer = imread(fullfile(unsmoothed_source, file_list{i}));
        init_z_stack(:,:,i) = rgb2gray(layer);    
    end 
    
    for j = 1:2000
        for k = 1:2000
            [Max, Index] = max(init_z_stack(j,k,:));
            binary_canvas(j,k,Index) = 1;
        end
    end
    
    for l = 1:num_classes
        image = binary_canvas(:,:,l);
        image = im2bw(image, 0.99);
        SE = strel('disk', 5);
        SE_2 = strel('disk', 10); 
        image = imclose(image, SE);
        image = imerode(image, SE_2);
        
        file_name = char(file_list(l));
        imwrite(image, fullfile(smoothed_target, file_name)); 
    end
        
end

smoothed_dir = dir(fullfile(smoothed_target));
smoothed_dir(1:2) = [];
smoothed_dir = natsortfiles({smoothed_dir.name});



%image_canvas = zeros(2000,2000,3);
%z_stack = zeros(2000,2000,13);

for jdir = 1:length(smoothed_dir)/num_classes
    
    image_canvas = zeros(2000,2000,3)+128;
    z_stack = zeros(2000,2000,num_classes);
    index = (jdir-1)*num_classes+1;
    file_names = smoothed_dir(index:index+num_classes-1);
    file_list = natsortfiles(file_names);
    file_one = file_list{1,1};
    file_root = extractBefore(file_one, "_class");
    
    %create layered probability z stack
    for i = 1:num_classes
        layer = imread(fullfile(smoothed_target, file_list{i}));
        z_stack(:,:,i) = layer;    
    end 
    
    %paint labels
    for j = 1:2000
        for k = 1:2000
            [Max, Index] = max(z_stack(j,k,:));
            if Max == 0
                continue
            end    
            image_canvas(j,k,:) = class_colors{Index};
        end
    end
    imwrite(uint8(image_canvas), fullfile(label_target, strcat(file_root, '_smoothed_label.png')));
    movefile(fullfile(proj_2_dir, strcat(file_root, '_input.png')), fullfile(label_target, strcat(file_root, '_input.png')));
end
















