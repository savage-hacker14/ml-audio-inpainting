% create_librispeech_dataset.m

% Pick 9 audio files from train-clean-360
librispeech_root_dir = "C:\Users\Jacob\Documents\2024\Northeastern\CS_6140\Project\LibriSpeech_360\train-clean-360";
filenames            = ["54-121080-0003.flac", "81-121543-0008.flac", "154-124003-0015.flac", ...
                        "434-132645-0008.flac", "464-126794-0036.flac", "510-130101-0034.flac", ...
                        "667-158816-0020.flac", "1012-133424-0012.flac", "1241-121103-0021.flac"];
audio_paths          = [];

% Create empty gap dataset table
n_files = length(filenames);
fs      = zeros(n_files, 1);
clean   = cell(n_files, 1);
mask80  = cell(n_files, 1);
gaps_table_librispeech = table(fs, clean, mask80, 'RowNames', filenames);

% Other parameters
audio_len_s = 5;
gap_start_s = 2;
gap_len_s   = 0.08;

% For each audio
for i=1:n_files
    % Determine full path
    subdirs  = split(filenames(i), '-');
    filepath = strcat(librispeech_root_dir, "/", subdirs(1), "/", subdirs(2), "/", filenames(i));

    % Load audio data
    [y, fs] = audioread(filepath);

    % Cut audio off at 5 seconds
    y = y(1:fs * audio_len_s);

    % Add a single 80ms gap at t = 2
    temp      = ones(fs * audio_len_s, 1);
    start_idx = fs * gap_start_s;
    end_idx   = fs * (gap_start_s + gap_len_s);
    temp(start_idx:end_idx) = 0;

    % Create mask: 1 for no gap, 0 for gap
    mask      = logical(temp);

    % Add data to table
    gaps_table_librispeech.fs(i)     = fs;
    gaps_table_librispeech.clean{i}  = y;
    gaps_table_librispeech.mask80{i} = mask;
end

% Save table data to .MAT file
save("gaps_table_librispeech.mat", "gaps_table_librispeech")