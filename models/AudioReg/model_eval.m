% model_eval.m
%
% This script loads in the reconstructed audio files

clc;

% Define file to use
filename            = "81-121543-0008.flac";
% addpath("../../test_samples_reconstructed/")

% Define models
models      = ["CNN_LSTM", "GAN", "AutoRegressive"];
n_models    = length(models);

% Create empty result table
model_names     = zeros(n_models, 1);
SDR             = zeros(n_models, 1);
PEMOQ           = zeros(n_models, 1);
PEAQ            = zeros(n_models, 1);
model_compare_table = table(SDR, PEMOQ, PEAQ, 'RowNames', models);

% Load in results from AutoRegressive model
load('results_09.mat');
load('gaps_table_librispeech.mat');
row_idx     = find(tables.janssen.signal == filename);
fs          = tables.janssen.fs(row_idx);
audio_len_s = 5.0;
gap_start_s = 2.0;
gap_len_s   = 0.08;

% Define mask: 80ms gap at t = 2s
temp      = ones(fs * audio_len_s, 1);
start_idx = fs * gap_start_s;
end_idx   = fs * (gap_start_s + gap_len_s);
temp(start_idx:end_idx) = 0;
mask      = logical(temp);  % 1 for no gap, 0 for gap

% Compute results
for i=1:n_models
    if (i ~= 3)
        % Deep Learning Models: Compute the metrics from the audio files
        % Load reconstructed file
        if (i == 1)
            [~, audio_name, ext] = fileparts(filename);
            filepath_orig    = strcat("../../test_samples/", audio_name, ext);
            filepath_reconst = strcat("../../test_samples_reconstructed/", audio_name, "_cnnlstm_inpainted", ext);
        elseif (i == 2)
            filepath_orig    = strcat("../../test_samples/", audio_name, ext);
            filepath_reconst = strcat("../../test_samples_reconstructed/", audio_name, "_gan_inpainted", ext);
        end

        % Load reconstructed audio
        [y, ~] = audioread(filepath_reconst);

        % Define signal and solution
        signal      = gaps_table_librispeech.clean{row_idx};
        solution    = y;

        % Compute SDR
        model_compare_table.SDR(i) = snr(signal(~mask), abs(signal(~mask)-solution(~mask)));      % Ensure SDR is positive
        figure; plot(signal(~mask)-solution(~mask));
        title(strcat('Model ', num2str(i)));

        % Compute PEMOQ
        model_compare_table.PEMOQ(i) = audioqual(signal, solution, fs);

        % Compute PEAQ - NOTE: Need to resample to 48 kHz
        original48  = resample(signal, 48000, fs);
        solution48  = resample(solution, 48000, fs);
        audiowrite("orig.wav", original48, 48000);
        audiowrite("solution.wav", solution48, 48000);
        model_compare_table.PEAQ(i) = PQevalAudio("orig.wav", "solution.wav", 0, length(solution));
        delete("orig.wav");
        delete("solution.wav");
    else
        % Auto-Regressive model: Copy results from MAT file
        model_compare_table.SDR(i)      = tables.janssen.SDR{row_idx, 1}(end);
        model_compare_table.PEMOQ(i)    = tables.janssen.PEMOQ{row_idx, 1}(end);
        model_compare_table.PEAQ(i)     = tables.janssen.PEAQ{row_idx, 1}(end);
    end
end

% Save model comparison table to MAT file
save("model_comparison.mat", "model_compare_table");