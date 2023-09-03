% Specify the folder containing the video files
videoFolder = 'D: dataset'; % Replace with the path to your folder

% Get a list of all video files in the folder
videoFiles = dir(fullfile(videoFolder, '*.avi'));

% Parameters
sigma = 5; % Width of the Gaussian distribution

% Create a directory to save averaged frames
outputDir = 'Apex_frames';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Loop through each video file in the folder
for fileIndex = 1:length(videoFiles)
    videoFile = fullfile(videoFolder, videoFiles(fileIndex).name);
    
    % Load the current video file
    video = VideoReader(videoFile);

    % Preallocate an array to store frames
    allFrames = uint8(zeros(video.Height, video.Width, 3, video.NumFrames));
    
    % Read all frames for the current video
    frameIndex = 1;
    while hasFrame(video)
        allFrames(:, :, :, frameIndex) = readFrame(video);
        frameIndex = frameIndex + 1;
    end
    
    % Compute Gaussian weight function based on the center frame
    centerFrameIndex = round(video.NumFrames / 2);
    gaussianWeights = exp(-((1:video.NumFrames) - centerFrameIndex).^2 / (2 * sigma^2));
    normalizedWeights = gaussianWeights / sum(gaussianWeights);

    % Apply Gaussian weights to all frames
    weightedFrames = zeros(size(allFrames));
    for i = 1:video.NumFrames
        % Compute Gaussian weights for this frame
        gaussianWeight = normalizedWeights(i);

        % Apply Gaussian weight to the current frame
        weightedFrames(:, :, :, i) = allFrames(:, :, :, i) * gaussianWeight;
    end

    % Compute the averaged frame
    averageFrame = uint8(sum(weightedFrames, 4));

    % Save the averaged frame
    imageName = sprintf('averaged_frame_%s.png', videoFiles(fileIndex).name);
    imwrite(averageFrame, fullfile(outputDir, imageName));
end

disp('Averaged frames saved successfully for all videos in the folder.');
