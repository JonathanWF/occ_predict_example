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
elseif isunix
    base_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'base_matlab');
    proj_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'occ_quant_risk_score');
else
    fprintf(1, 'Unknown filesystem, please edit folder setup!\n');
    return;
end

% Add stuff to the path
pathCell = regexp(path, pathsep, 'split');
if ispc
    base_dir_onPath = any(strcmpi(base_dir, pathCell));
    proj_dir_onPath = any(strcmpi(proj_dir, pathCell));
else
    base_dir_onPath = any(strcmp(base_dir, pathCell));
    proj_dir_onPath = any(strcmp(proj_dir, pathCell));
end
if ~base_dir_onPath
    fprintf(1, 'Adding base_dir to path\n');
    addpath(genpath(base_dir));
end
if ~proj_dir_onPath
    fprintf(1, 'Adding proj_dir to path\n');
    addpath(genpath(fullfile(proj_dir, 'scripts')));
    addpath(genpath(fullfile(proj_dir, 'module')));
end

% Slice Sources
stitched_source = fullfile(proj_dir, 'data', 'TNT stitched');
stitched_dir = dir(fullfile(stitched_source));
target_dir = fullfile(proj_dir, 'data', 'tumor non tumor images');

%Load spreadsheet
image_info = readtable(fullfile(proj_dir, 'data', 'histomics_data_tnt_processing.csv'));
image_names = image_info(:,3);
image_info = image_info(:,9:17);
image_info = sortrows(image_info, 9);
stitched_dir(1:2) = [];
image_info = table2array(image_info);
image_names = table2array(image_names);

for idir = 1:length(image_names)
    name = string(image_names(idir,:));
    original_image = imread(fullfile(stitched_source, strcat(name, '.tifforiginal.png')));
    label_image = imread(fullfile(stitched_source, strcat(name, '.tiffsmoothed_label_version4TNT.png')));
    scale_factor = (image_info(idir, 5)*4)*0.086311589174185;
    scale_top = floor(image_info(idir,1)*scale_factor);
    scale_left = floor(image_info(idir,2)*scale_factor);
    scale_width = floor(image_info(idir,3)*scale_factor);
    scale_height = floor(image_info(idir,4)*scale_factor);
    original_tile = imcrop(original_image, [scale_left scale_top scale_width scale_height]);
    label_tile = imcrop(label_image, [scale_left scale_top scale_width scale_height]);
    [x, y, z] = size(label_tile);
    check_prelim = unique(label_tile);
   
    for i = 1:x
        for j = 1:y
            if label_tile(i,j,1)== 0 && label_tile(i,j,2) == 0 && label_tile(i,j,3) == 255
                continue
            else
                label_tile(i,j,:) = [0,0,0];
            end
        end
    end
    imwrite(original_tile, fullfile(target_dir, strcat(name, 'original_cropped.png')));
    imwrite(label_tile, fullfile(target_dir, strcat(name, 'tumor_nontumor_label_version4TNT.png')));
end
