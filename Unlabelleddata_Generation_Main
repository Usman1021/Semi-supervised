
% Specify the folder containing the video files
videoFolder = 'G:datasets'; % Replace with the path to your folder

% Get a list of all video files in the folder
videoFiles = dir(fullfile(videoFolder, '*.mov'));

% Parameters
segmentSize = 80;  % Number of frames per segment
sigma = 5;         % Width of the Gaussian distribution

% Create a directory to save averaged frames
outputDir = '80';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Loop through each video file in the folder
for fileIndex = 1:length(videoFiles)
    videoFile = fullfile(videoFolder, videoFiles(fileIndex).name);
    
    % Load the current video file
    video = VideoReader(videoFile);

    % Calculate total number of segments
    numSegments = floor(video.NumFrames / segmentSize);

    % Compute Gaussian weight functions and average frames for each segment
    for segIndex = 1:numSegments
        startFrame = (segIndex - 1) * segmentSize + 1;
        endFrame = startFrame + segmentSize - 1;

        % Ensure endFrame does not exceed the total number of frames
        if endFrame > video.NumFrames
            endFrame = video.NumFrames;
        end

        % Read frames for the current segment
        segmentFrames = read(video, [startFrame, endFrame]);

        % Compute Gaussian weight function
        centralFrameIndex = round((startFrame + endFrame) / 2);
        gaussianWeights = exp(-((1:segmentSize) - centralFrameIndex).^2 / (2 * sigma^2));
        normalizedWeights = gaussianWeights / sum(gaussianWeights);

        % Apply Gaussian weights to the frames
        weightedFrames = zeros(size(segmentFrames));
        for i = 1:segmentSize
            weightedFrames(:, :, :, i) = segmentFrames(:, :, :, i) * normalizedWeights(i);
        end

        % Compute the averaged frame
        averageFrame = uint8(sum(weightedFrames, 4));

        % Save the averaged frame
        imageName = sprintf('averaged_frame_%s_%03d.png', videoFiles(fileIndex).name, segIndex);
        imwrite(averageFrame, fullfile(outputDir, imageName));
    end
end

disp('Averaged frames saved successfully for all videos in the folder.');
