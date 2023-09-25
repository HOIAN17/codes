rng('shuffle');

%% set up
clear;
close all;
clc;

%% Hyperparameter
hyperBlockSize = 120; %.% the number of total trials during a run
hyperWaiting = 3850/1000;
hyperFixation = 300/1000;
hyperStimulus = 500/1000;
TR = input('Repetition time (TR) : ','s'); % ITIs will be randomly selected based on TR.
if TR == 1.5
    disp('TR = 1.50 sec.')
    hyperITI = [2200, 2200, 2200, 3750]/1000;
else
    disp('TR = 1.25 sec.')
    hyperITI = [1700, 1700, 1700, 1700, 2950, 2950, 4200]/1000; 
end

%% Subject ID input
sub_id = input('subject id: 00 format :','s');
sub_id = ['sub-' sub_id];

%% Load stimuli : Make a stimuliFolder struct that contains scences with filenames
addpath(genpath('stimuli'));
stim = 'object';
stimFolder = dir(fullfile('stimuli'));
stimul = {stimFolder(:).name};
stimulF.(stim) = stimul;
stimulF.(stim)(1) = [];
stimulF.(stim)(1) = [];

%% Build a design matrix
WM_R = zeros(600,4); % Result
% column 1 : Image - 1 Bicycle 2 Bird 3 Dress 4 Face 5 House
% column 2 : Task condition 1 flipped 2 nonflipped
% column 3 : Task response (or pressed button) - 1 press 2 no press
% column 4 : Reaction time - float

images = [1,2,3,4,5];
allSequences = perms(images);

shuffledSequences = allSequences(randperm(size(allSequences,1)),:)';
WM_R(:,1) = shuffledSequences(:);

flippedSequences = floor(allSequences/5)+1;
reshapedSequences2 = flippedSequences(randperm(size(flippedSequences,1)),:)';
WM_R(:,2) = reshapedSequences2(:);

%% Set screen

Screen('Preference', 'SkipSyncTests', 1); %.% 모니터 주사율 이슈를 스킵하겠다.
PsychDefaultSetup(2);

screens = Screen('Screens');
screenNumber = max(screens);
% screenNumber = 1; %.% To check in my PC monitor

white = 255*WhiteIndex(screenNumber);
black = 255*BlackIndex(screenNumber);
gray = white / 2;

[window, windowRect] = Screen('OpenWindow', screenNumber, gray);
topPriorityLevel = MaxPriority(window);
[xCenter, yCenter] = RectCenter(windowRect);
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

%% set key code
flip = KbName('1');
sync = KbName('s');

RestrictKeysForKbCheck([flip sync]);

%% waiting for start
% present fixation
tic;
% HideCursor;

Screen('FillRect', window, gray);
Screen('TextSize', window, 25);
DrawFormattedText(window, 'PRESS 1 only for FLIP', 'center', 'center', black);
Screen('Flip', window);

WaitSecs(3.85);

%% Main loop
rect = [330 330]; % stimulus size

for p = 1:hyperBlockSize
    if rem(p, hyperBlockSize) == 1
        DrawFormattedText(window, 'if you want to finish break time, press SPACE BAR', 'center', 'center', black);
        Screen('Flip', window);
    
        done = 0;
        while(~done)
            [KeyIsDown, Secs, KeyCode] = KbCheck;
            if KeyIsDown
                if KeyCode(finish)
                    done = 1;
                end
            end
        end
    end

    %% Waiting Period for the first trial (3850ms)
    if rem(p, hyperBlockSize) == 1
        Screen('FillRect', window, gray);
        wvbl = Screen('Flip', window);
        timerStart = GetSecs %TIME%
    end

    %% Present fixation (300ms)
    Screen('FillRect', window, gray);
    DrawFormattedText(window, '+', 'center', 'center', 255*[1 1 1]);
    if rem(p, b) == 1
        fvbl = Screen('Flip', window, wvbl+hyperWaiting);
    else
        fvbl = Screen('Flip', window);
    end

    %% Present image (500ms)
    Screen('FillRect', window, gray);
    targetP = imread(stimulF.(stim){WM_R(p,1)});
    targetP = rgb2gray(targetP);
    targetP = imresize(targetP, rect);
    Ptarget = Screen('MakeTexture', window, targetP);

    if WM_R(p,2) == 1
        Screen('DrawTexture', window, Ptarget, [], [], 0);
    else
        Screen('DrawTexture', window, Ptarget, [], [], 0, rotate180)
    end

    DrawFormattedText(window, '+', 'center', 'center', black); %.% Por que este es aqui?
    
    %.% if S key
    vbl = Screen('Flip', window, fvbl+hyperFixation); % Right after 300ms fixation
    
%     send_trigger(daq_id, 0, 6, SADI);
    disp(['this condition : ' checkCond]) %.% Is it ok?

    % check WM reaction|save information
    done = 0;
    while(~done)
        [KeyIsDown, Secs, KeyCode] = KbCheck;
        if KeyIsDown
            if KeyCode(flip)
                WM_R(p,3) = 1; % response
                WM_R(p,4) = Secs - vbl; % action
            end
        end
    
        if Secs - vbl > 0.4
            done = 1;
        end
    end

    Screen('FillRect', window, gray);
%     DrawFormattedText(window, '+', 'center', 'center', black); % change.. think..
    vbl2 = Screen('Flip', window);

    % check WM reaction|save information
    iti = hyperITI(randi(length(hyperITI)));
    done = 0;
    while(~done)
        [KeyIsDown, Secs, KeyCode] = KbCheck;
        if KeyIsDown
            if KeyCode(flip)
                WM_R(p,3) = 1; % response
                WM_R(p,4) = Secs - vbl; % action
                disp('the button is pressed !')
            end
        end

        if Secs - vbl2 > iti
            done = 1;
        end
    end
  
end

while done
    if end_time - start_time == 0
        done = 0;
    end
end

sca;
% ShowCursor;
toc;

%% save design matrix and working memory response
% check and make data file
if ~isfolder(fullfile(pwd,'deutschResult',sub_id))
    mkdir(fullfile(pwd,'deutschResult',sub_id));
end

% set file name
wm_file = fullfile(pwd,'deutschResult',sub_id,[sub_id '_deutschResult_behavior.mat']);

% save file
save(wm_file, 'WM_R');

load handel;
sound(y,Fs);
