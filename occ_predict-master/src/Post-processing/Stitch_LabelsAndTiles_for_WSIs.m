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
unstitched_source = fullfile(proj_dir, 'data', 'TNT stitching');
unstitched_dir = dir(fullfile(unstitched_source));

%Load spreadsheet
image_info = readmatrix(fullfile(proj_dir, 'data', 'histomics_data_tnt_processing.csv'));
image_info = image_info(:,9:17);
image_info(isnan(image_info(:,1:4))) = 0;
image_info = sortrows(image_info, 9);
image_info(any(isnan(image_info),2),:) = [];
unstitched_dir(1:2) = [];

% Binary Files Saved here
stitched_dir = fullfile(proj_dir, 'data', 'TNT stitched');




for idir = 1:length(unstitched_dir)
    
    slice_number = unstitched_dir(idir).name;
    unstitched_dir_selection = fullfile(unstitched_source, slice_number);
    
    %image size stuff
    image_rows = image_info(idir,7);
    image_columns = image_info(idir,8);
    scale_table = image_info(idir,5);
    scale_factor = 0.086311589174185*scale_table*4;
    image_rows_scaled = ceil(image_rows*scale_factor);
    image_columns_scaled = ceil(image_columns*scale_factor);
    num_tiles_rows = ceil(image_rows_scaled/1990);
    num_tiles_columns = ceil(image_columns_scaled/1990);

    row_padding = 1990-(image_columns_scaled-(1990*(num_tiles_columns-1)));
    column_padding = 1990-(image_rows_scaled-(1990*(num_tiles_rows-1)));
    dummy_tile = zeros(1990, 1990, 3, 'uint8');
    dummy_label = dummy_tile+255;

    num_tiles_rows = ceil(image_columns_scaled/1990);
    num_tiles_columns = ceil(image_rows_scaled/1990);
    full_original_image = zeros(image_columns_scaled, image_rows_scaled, 3, 'uint8');
    full_label_image = zeros(image_columns_scaled, image_rows_scaled, 3, 'uint8');
    
    unstitched_cell_dir = dir(unstitched_dir_selection);
    unstitched_cell_dir(1:2) = [];
    file_names = {unstitched_cell_dir(:).name};
    file_list = natsortfiles(file_names);

    file_one = file_list{1,1};
    file_root = extractBefore(file_one, "_");

    %columns
    for i = 1:num_tiles_rows
       row_name = 1+(i-1)*(23056/(scale_table*4))-1;
       %rows 
       for j = 1:num_tiles_columns
            column_name = 1+(j-1)*(23056/(scale_table*4))-1; 
            filename = fullfile(unstitched_dir_selection, strcat(file_root, "_row_", string(row_name), ".0_col_", string(column_name), ".0_input.png"));
            label_filename = strrep(filename, "input", "smoothed_label");
            row_number = 1990*(i-1)+1;
            column_number = 1990*(j-1)+1;
            if isfile(filename)
                if (i == num_tiles_rows) && (j == num_tiles_columns)
                    image_tile = imread(filename);
                    image_tile_cropped = imcrop(image_tile, [6 6 1990-column_padding-1 1990-row_padding-1]);
                    label_tile = imread(label_filename);
                    label_tile_cropped = imcrop(label_tile, [6 6 1990-column_padding-1 1990-row_padding-1]);
                    full_original_image(row_number:row_number+1989-row_padding, column_number:column_number+1989-column_padding,  1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989-row_padding, column_number:column_number+1989-column_padding, 1:3) = label_tile_cropped;
                elseif (i == num_tiles_rows)
                    image_tile = imread(filename);
                    image_tile_cropped = imcrop(image_tile, [6 6 1990-1 1990-row_padding-1]);
                    label_tile = imread(label_filename);
                    label_tile_cropped = imcrop(label_tile, [6 6 1990-1 1990-row_padding-1]);
                    full_original_image(row_number:row_number+1989-row_padding, column_number:column_number+1989, 1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989-row_padding, column_number:column_number+1989, 1:3) = label_tile_cropped;
                elseif (j == num_tiles_columns)
                    image_tile = imread(filename);
                    image_tile_cropped = imcrop(image_tile, [6 6 1990-column_padding-1 1990-1]);
                    label_tile = imread(label_filename);
                    label_tile_cropped = imcrop(label_tile, [6 6 1990-column_padding-1 1990-1]);
                    full_original_image(row_number:row_number+1989, column_number:column_number+1989-column_padding, 1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989, column_number:column_number+1989-column_padding, 1:3) = label_tile_cropped;
                else
                    image_tile = imread(filename);
                    image_tile_cropped = imcrop(image_tile, [6 6 1990-1 1990-1]);
                    label_tile = imread(label_filename);
                    label_tile_cropped = imcrop(label_tile, [6 6 1990-1 1990-1]);
                    full_original_image(row_number:row_number+1989, column_number:column_number+1989,  1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989, column_number:column_number+1989, 1:3) = label_tile_cropped;
                end
            else
                if (i == num_tiles_rows) && (j == num_tiles_columns)
                    image_tile = dummy_tile;
                    image_tile_cropped = imcrop(image_tile, [1 1 1990-column_padding-1 1990-row_padding-1]);
                    label_tile = dummy_label;
                    label_tile_cropped = imcrop(label_tile, [1 1 1990-column_padding-1 1990-row_padding-1]);
                    full_original_image(row_number:row_number+1989-row_padding, column_number:column_number+1989-column_padding, 1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989-row_padding, column_number:column_number+1989-column_padding, 1:3) = label_tile_cropped;
                elseif (i == num_tiles_rows)
                    image_tile = dummy_tile;
                    image_tile_cropped = imcrop(image_tile, [1 1 1990-1 1990-row_padding-1]);
                    label_tile = dummy_label;
                    label_tile_cropped = imcrop(label_tile, [1 1 1990-1 1990-row_padding-1]);
                    full_original_image(row_number:row_number+1989-row_padding, column_number:column_number+1989,  1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989-row_padding, column_number:column_number+1989,  1:3) = label_tile_cropped;
                elseif (j == num_tiles_columns)
                    image_tile = dummy_tile;
                    image_tile_cropped = imcrop(image_tile, [1 1 1990-column_padding-1 1990-1]);
                    label_tile = dummy_label;
                    label_tile_cropped = imcrop(label_tile, [1 1 1990-column_padding-1 1990-1]);
                    full_original_image(row_number:row_number+1989, column_number:column_number+1989-column_padding,  1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989, column_number:column_number+1989-column_padding,  1:3) = label_tile_cropped;
                else
                    image_tile = dummy_tile;
                    image_tile_cropped = imcrop(image_tile, [1 1  1990-1 1990-1]);
                    label_tile = dummy_label;
                    label_tile_cropped = imcrop(label_tile, [1 1  1990-1 1990-1]);
                    full_original_image(row_number:row_number+1989, column_number:column_number+1989,  1:3) = image_tile_cropped;
                    full_label_image(row_number:row_number+1989, column_number:column_number+1989,  1:3) = label_tile_cropped;
                end
            end
       end
    end
    save_root = strcat(file_root, "original.png");
    imwrite(full_original_image, fullfile(stitched_dir, save_root));
    imwrite(full_label_image, fullfile(stitched_dir, strrep(save_root, "original", "smoothed_label_version4TNT")));
end



