% This example shows how to solve the forward problem and the inverse problem
% in a three-dimensional numerical model.
% A 3D sensor with 32 electrodes arranged in two rings of 16 electrodes each is used.

%clear all;
%close all;

addpath fwdp / % Folder containing files required for model preparation and forward problem computation
addpath invp / % Folder containing files required for image reconstuction
addpath visualization / % Folder containing files required for visualization of the results

%% Defining parameters of the model
% inf / 50 objs with 64x64x64 decim 1 does not work do not use
% 500s / 100 objs with 64x64x64 decim 2 (5s)
% 186s / 16 objs with 128x128x64 decim 4 (11.6s)
workspaceSize = 160; % Size of workspace in mm
meshSize = 64; % Size of meshgrid in pixels. The number of pixels must be a power of two
modelMeshHeight = 64;
model_meshing = 2;
model = defineWorkspace(workspaceSize, workspaceSize, workspaceSize); % Order of specifying values: (x-axis, y-axis, z-axis)
model = defineMesh(model, meshSize, meshSize, modelMeshHeight); % The number of pixels can vary across different axes but must each be a power of two
model.f = 1e6; % Frequency of the excitation signal used in the measurement

%% Permittivity and conductivity values of materials
const.metal_eps = 1; % Relative permittivity of the material
const.metal_sigma = 1e7; % Absolut conductivity of the material
const.air_eps = 1;
const.air_sigma = 1e-15;
const.pmma_eps = 3;
const.pmma_sigma = 1e-21;
const.pvc_eps = 2;
const.pvc_sigma = 1e-21;

%% Creating objects that will be used in the simulation
model = newSimpleElement(model, 'cuboid', 'all', [0, 0, 0], [0, 0, 0], [workspaceSize, workspaceSize, workspaceSize]); % Whole workspace
model = makeTomograph(model, workspaceSize / 2, workspaceSize, 2, 2, 32, [-35 35], 77.5, 50, 16, 1.6, const);
model = newSimpleElement(model, 'cylinder', 'fovcenter', [0, 0, 0], [0, 0, 0], [73, 73, workspaceSize]);

model.voltage_range = 10; % Amplitude of excitation signal [V]
model.measurements_all = 1; % ...

%% generate min and max models
modelMinMax = false;

if modelMinMax
    %% Preparing three models: for an empty sensor (min), a full sensor (max), and a sensor with a test object (obj)
    modelMin = model;
    modelMin = addElement(modelMin, 'fov0', const.air_eps, const.air_sigma);

    modelMax = model;
    modelMax = addElement(modelMax, 'fov0', const.pvc_eps, const.pvc_sigma);

    %% Obtaining maps of electrical parameter distributions in each model

    modelMin.eps_map = getPermittivityMap(modelMin);
    modelMin.sigma_map = getConductivityMap(modelMin);
    modelMin = boundaryVoltageVector(modelMin);
    [modelMin.patternImage] = getElementMap(modelMin);

    modelMax.eps_map = getPermittivityMap(modelMax);
    modelMax.sigma_map = getConductivityMap(modelMax);
    modelMax = boundaryVoltageVector(modelMax);
    [modelMax.patternImage] = getElementMap(modelMax);

    %% Setting up the discretization grid

    modelMin = fineMesh(modelMin, 'denseMeshArea', 1);
    modelMin = meshing(modelMin, 1, 4);

    modelMax = fineMesh(modelMax, 'denseMeshArea', 1);
    modelMax = meshing(modelMax, 1, 4);

    %% Calculating potential and electric field distributions in each model

    modelMin = calculateElectricField(modelMin);
    modelMax = calculateElectricField(modelMax);

    %% Calculating sensitivity matrices for each model

    modelMin = calculateSensitivityMaps(modelMin);
    modelMax = calculateSensitivityMaps(modelMax);

    %% Simulating measured capacitances

    modelMin.C = modelMin.qt.Sens(:, :).' * (model.eps0 * modelMin.qt.eps(:) + 1i * (modelMin.qt.sigma(:) * 2 * pi * model.f));
    modelMax.C = modelMax.qt.Sens(:, :).' * (model.eps0 * modelMax.qt.eps(:) + 1i * (modelMax.qt.sigma(:) * 2 * pi * model.f));
end

%% generate training data parameters
nObj = 100;

calculate = true;
disp = false;

% triangle distribution
lower = 1;
peak = 10;
upper = 20;
%pd = makedist('Triangular', 'A', lower, 'B', peak, 'C', upper);
pd = makedist('Uniform', 'lower', lower, 'upper', upper);

% other params
maxSize = 77;
minSize = 5;
minAngl = 10;
shapes = ["cuboid", "ellipsoid", "cylinder"];

prob_hollow = 0;
prob_cut = 0.5;

% filePrefix = 'd:\3d_ect_data\3d_random_shapes';
filePrefix = '.\temp\3d_random_shapes';
fieldsToSave = {'C', 'eps_map'};


%% calculation start
[fov] = findElement('fov0', model.Elements);

tStart = tic;
delete(gcp('nocreate'))
pool = parpool("Threads");

% change to parfor if more than 1 object and you dont neet to plot!!!
parfor sim_number = 1:nObj
%for sim_number = 1:nObj

    modelObj = model;

    objects = {};

    fov_string = 'fov0';

    % generate number of objects
    r = random(pd);
    shape = randi([1 length(shapes)], 1, floor(r));

    for i = 1:r
        o_name = ['o' num2str(i)];
        retry = [];

        while 1
            o_name = [o_name retry];
            angl = randi([0, 360], 1, 3);

            while 1
                pos = randi([-maxSize maxSize], 1, 3);
                % dim = randi([minSize max(round(maxSize / nthroot(r, 3)), minSize)], 1, 3);
                dim = randi([minSize maxSize], 1, 3);
                % check if position and size is smaller than tomoghraph
                if max(abs(pos(:)) + max(dim)) < maxSize
                    break;
                end

            end

            % decide on cutaway
            x = rand;

            if x > prob_cut
                cut = [0 360];
            else

                while 1
                    cut = randi([0, 360], 1, 2);

                    if abs(cut(1) - cut(2)) > minAngl
                        break;
                    end

                end

            end

            x = rand;

            if x > prob_hollow
                hollow = 0;
            else
                hollow = randi([0, min(dim)]);
            end

            if cut(1) > cut(2)
                cut(1) = cut(1) - 360;
            end

            modelObj = newSimpleElement(modelObj, shapes(shape(i)), o_name, pos, angl, dim, cut, hollow);

            % check intersection with fov
            [elm] = findElement(o_name, modelObj.Elements);
            [sect, ~, ~] = intersect(modelObj.Elements{elm}.location_index, modelObj.Elements{fov}.location_index);

            if size(modelObj.Elements{elm}.location_index, 1) == size(sect, 1) % true if fov and elm full overlap
                % check other object intersection
                overlap = {};
                overlapIndex = {};
                for j = 1:length(objects)
                    if isempty(objects{j})
                        continue;
                    end
                    [elm2] = findElement(objects{j}, modelObj.Elements);
                    [sect, ~, ~] = intersect(modelObj.Elements{elm}.location_index, modelObj.Elements{elm2}.location_index);

                    if ~isempty(sect) % if tere is overlap combine
                        overlap{end+1} = objects{j};
                        overlapIndex{end+1} = j;
                    end

                end

                if isempty(overlap) || isempty(objects) % add object if not added into complex or is first object
                    objects{end + 1} = o_name;
                else
                    new_name = [];
                    new_obj_formula = [];
                    for o = 1:length(overlap)
                        new_name = [new_name overlap{o}];
                        new_obj_formula = [new_obj_formula overlap{o}, '+'];
                    end
                    new_name = [new_name o_name];
                    new_obj_formula = [new_obj_formula o_name];

                    modelObj = newComplexElement(modelObj, new_name, new_obj_formula);
                    objects{end+1} = new_name;

                    for o = 1:length(overlapIndex)
                        objects{overlapIndex{o}} = [];
                    end
                end

            else % does not full overlap with fov, regenerate object
                retry = [retry 'r'];
                continue;
            end

            break; % should only get here if no retry
        end

    end

    % add all elements
    for i = 1:length(objects)
        if isempty(objects{i})
            continue;
        end
        modelObj = addElement(modelObj, objects{i}, const.pmma_eps, const.pmma_sigma);
        fov_string = [fov_string, '-', objects{i}];
    end

    % fov_string = strjoin(fov_string);

    % add fov
    modelObj = newComplexElement(modelObj, 'fov', fov_string); % FOV beside objects
    modelObj = addElement(modelObj, 'fov', const.air_eps, const.air_sigma);

    % simulate
    modelObj.eps_map = getPermittivityMap(modelObj);
    modelObj.sigma_map = getConductivityMap(modelObj);
    modelObj = boundaryVoltageVector(modelObj);
    [modelObj.patternImage] = getElementMap(modelObj);
    
    %% test object inside sensor generation done

    if disp
        drawPatternImage(modelObj, 'surf');
    end

    if calculate
        modelObj = fineMesh(modelObj, 'denseMeshArea', 1);
        modelObj = meshing(modelObj, 1, model_meshing);
        modelObj = calculateElectricField(modelObj);
        modelObj = calculateSensitivityMaps(modelObj);

        % todo check if this is correct !!!!!!!!!!!
        modelObj.C = real(modelObj.qt.Sens(:,:).'*modelObj.eps0*(modelObj.qt.eps(:)+1i*modelObj.qt.sigma(:)));
        % !!!!!!!!!!!!!!!!!!!!

        if disp
            %drawElectricField(modelObj,'mm','real','Em',8);
            drawPotential(modelObj,'slice','real',16);
        end

        % write data to file
        fname = [filePrefix '_' num2str(sim_number) '_' num2str(randi(2 ^ 32)) '.mat'];

        % todo what do we need
        save(fname, '-fromstruct', modelObj, fieldsToSave{:}, "-nocompression");
        fprintf([fname ' saved']);
    end
end

delete(pool);
fprintf(' . Done. '); toc(tStart)
