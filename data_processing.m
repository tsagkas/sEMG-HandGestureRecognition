%% Training signals - creation - Myo - step #1
clc;
clear all;

START = 's';
END = 'e1.mat';
filename = "s";

for subject = 1 : 8
    %_________________________________________
    code = strcat(START,int2str(subject),END);
    disp(code);

    struct = load(code);
    rep = struct.pulse;
    emg = struct.emg;
   %_________________________________________

    counter = 1;
    points = zeros(1, 50);
    
    for i = 2 : length(rep)
        if(rep(i - 1) == 0 && rep(i) > 0)
            points(counter) = i;
            counter = counter + 1;
        end
        if(rep(i - 1) > 0 && rep(i) == 0)
            points(counter) = i - 1;
            counter = counter + 1;
        end 
    end
    
   
    counter = 1;
    repeat = 0;
    exercise = 0;
    exe_number = 1;
    
    for i = 1 : length(points) / 2
        starting_point = points(counter);
        counter = counter + 1;
        finishing_point = points(counter);
        counter = counter + 1;
        
         disp(starting_point)
         disp(finishing_point)
         disp('______________')
        
        repeat = repeat + 1;
        exercise = exercise + 1;
        
        columns = finishing_point - starting_point + 1;
        matrix = zeros(8, columns);
         
        column = 1;
        for pointer = starting_point : finishing_point
            for channel = 1 : 8
                matrix(channel, column) = emg(pointer, channel);
            end
            column = column + 1;
        end
         
        file = filename + subject + "e" + exe_number + "rep" + repeat + ".mat";
        save(file, 'matrix')
        
        if(repeat == 5)
            repeat = 0;
        end
        if(exercise == 5)
            exercise = 0;
            exe_number = exe_number + 1;
        end
    end
end   


%% Maximum Size Matrix - step #2

clc;

maximum_size = 0;

emg_sizes = zeros(1, 3400);
counter = 0;

for subject = 1 : 8
    for exercise = 1 : 5
        for repetition = 1 : 5
            
            counter = counter + 1;
             
            filename = strcat('s', int2str(subject), 'e', int2str(exercise), 'rep', int2str(repetition));
            path = 'Myo_training\E1\';
            
            load_path = strcat(path, filename);
            load_path = strcat(load_path, '.mat');
            
            signal = load(load_path); 
            matrix = signal.matrix;
            
            matrix_size = size(matrix); matrix_size = matrix_size(2);
            emg_sizes(1, counter) = matrix_size;
            
            if(matrix_size > maximum_size)
                maximum_size = matrix_size;
            end
        end
    end
end

%% Training Images - creation II - step #3

clear all;
clc;

for subject = 1 : 8
    disp('Subject: ');
    disp(subject)
    for exercise = 1 : 12
        for repetition = 1 : 5
            
            filename = strcat('s', int2str(subject), 'e', int2str(exercise), 'rep', int2str(repetition));
            path = 'Myo_training\E2\';
            
            load_path = strcat(path, filename);
            load_path = strcat(load_path, '.mat');
            
            signal = load(load_path); 
            
            original_matrix = signal.matrix;
            noisy_matrix = awgn(original_matrix,25);
            
            matrix_size = size(original_matrix); matrix_size = matrix_size(2);
            window_type_one = matrix_size / 6;
            window_type_one = int16(fix(window_type_one));
            window_type_two = 226 - window_type_one;
            
            if(window_type_one > 226)
                window_type_one = 225;
            end
            
            start_index = 1;
            end_index = 15;

%________%: Normal Windows.    
            
            for div = 1 : window_type_one
                if(exercise ~= 10 && exercise ~= 11 && exercise ~= 12)
                    filename = strcat('X2_e0', int2str(exercise), '_s', int2str(subject), '_E2_rep', int2str(repetition));
                    fn = strcat('X2_e0', int2str(exercise), '_s', int2str(subject), '_E2_rep', int2str(repetition));
                else
                    filename = strcat( 'X2_e', int2str(exercise), '_s', int2str(subject), '_E2_rep', int2str(repetition));
                    fn = strcat( 'X2_e', int2str(exercise), '_s', int2str(subject), '_E2_rep', int2str(repetition));
                end

                counter = 1;
                
                if(end_index <= matrix_size) 
                    filename = strcat(filename, '_image', int2str(div));

                    if(repetition == 1 || repetition == 3 || repetition == 5)
                        save_matrix_path = strcat('EMG_data\train_set\' , filename, '.mat');
                        save_matrix_path_gaussian = strcat('EMG_data\train_set\' , filename, '_GAUSSIAN.mat'); 
                    end

                    if(repetition == 2)
                        save_matrix_path = strcat('EMG_data\test_set\' , filename, '.mat');
                        save_matrix_path_gaussian = strcat('EMG_data\test_set\' , filename, '_GAUSSIAN.mat');
                    end

                    if(repetition == 4)
                        save_matrix_path = strcat('EMG_data\val_set\' , filename, '.mat');
                        save_matrix_path_gaussian = strcat('EMG_data\val_set\' , filename, '_GAUSSIAN.mat'); 
                    end

                    image = zeros(8, 15);
                    image = original_matrix(1:8, start_index : end_index);
                    
                    save(save_matrix_path, 'image')
                    
                    image = zeros(8, 15);
                    image = noisy_matrix(1:8, start_index : end_index);

                    save(save_matrix_path_gaussian, 'image')
                    
                    start_index = start_index + 6;
                    end_index = end_index + 6;
                else
                    div = div - 1;
                end
            end

%________%: Random Windows.

            for index = div : 226
                
                limit = matrix_size - 20;
                random_number = randi([1,limit]);
                
                index_start_random = random_number;
                index_end_random = index_start_random + 14;

                if(repetition == 1 || repetition == 3 || repetition == 5)
                    save_path = strcat('EMG_data\train_set\' , fn, '_image', int2str(index), '.mat');
                    save_path_gaussian = strcat('EMG_data\train_set\' , fn, '_image', int2str(index), '_GAUSSIAN.mat');
                end
                
                if(repetition == 2)
                    save_path = strcat('EMG_data\test_set\' , fn, '_image', int2str(index), '.mat');
                    save_path_gaussian = strcat('EMG_data\test_set\' , fn, '_image', int2str(index), '_GAUSSIAN.mat');
                end
                
                if(repetition == 4)
                    save_path = strcat('EMG_data\val_set\' , fn, '_image', int2str(index), '.mat');
                    save_path_gaussian = strcat('EMG_data\val_set\' , fn, '_image', int2str(index), '_GAUSSIAN.mat');
                end
                   
                image = zeros(8, 15); 
                
                image = original_matrix(1:8, index_start_random : index_end_random);
                save(save_path, 'image');
                
                image = zeros(8, 15);
                
                image = noisy_matrix(1:8, index_start_random : index_end_random);
                save(save_path_gaussian, 'image')
                
            end
        end
    end
end