"""

@author: bin.xue
Based on works of chkarada.
Add step to generate the file list and modify the noisy file name.

"""

# Note: This single process audio synthesizer will attempt to use each clean
# speech sourcefile once, as it does not randomly sample from these files

import os
import sys
import glob
import argparse
import configparser as CP
from random import shuffle
import random
import multiprocessing
from multiprocessing import Queue, Process

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.io import wavfile

from audiolib import audioread, audiowrite, segmental_snr_mixer, activitydetector, is_clipped
import utils
import logging

MAXTRIES = 50
MAXFILELEN = 100

#np.random.seed(5)
#random.seed(5)

num_processes = min(os.cpu_count(), 64)  # 最大16进程  # Number of processes to use


def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0: clean_speech.shape[0]]

    return reverb_speech

"""
# 引入 functools.lru_cache 优化缓存机制
from functools import lru_cache

@lru_cache(maxsize=10)
def audioread_with_cache(file_path):
    # 使用当前进程的logger实例
    process_logger = logging.getLogger(f"Process-{multiprocessing.current_process().pid}")
    process_logger.info(f"Loading audio file: {file_path}")  # 使用进程专属logger
    audio, fs = audioread(file_path)
    return audio, fs
"""

def build_audio(is_clean, params, index, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])

    output_audio = np.zeros(audio_samples_length)  # 预分配数组空间
    current_pos = 0
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        source_files = params['cleanfilenames']
        idx = index
    else:
        if 'noisefilenames' in params.keys():
            source_files = params['noisefilenames']
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params['noisedirs']
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(os.path.join(noisedirs[idx_n_dir],
                                                  params['audioformat']))
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output * silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files)
        #input_audio, fs_input = audioread_with_cache(source_files[idx])
        process_logger = logging.getLogger(f"Process-{multiprocessing.current_process().pid}")
        process_logger.info(f"Loading audio file: {source_files[idx]}")  # 使用进程专属logger
        input_audio, fs_input = audioread(source_files[idx])
        if input_audio is None:
            sys.stderr.write("WARNING: Cannot read file: %s\n" % source_files[idx])
            continue
        if fs_input != fs_output:
            #input_audio = librosa.resample(input_audio, fs_input, fs_output)
            input_audio = librosa.resample(input_audio, orig_sr=fs_input, target_sr=fs_output)

        # 修改多进程版本 build_audio 函数的音频处理部分
        segment_len = min(len(input_audio), remaining_length)

        # 添加随机截取逻辑（与单进程版本一致）
        if len(input_audio) > segment_len and (not is_clean or not params['is_test_set']):
            idx_seg = np.random.randint(0, len(input_audio) - segment_len)
            input_segment = input_audio[idx_seg:idx_seg + segment_len]
        else:
            input_segment = input_audio[:segment_len]


        # check for clipping, and if found move onto next file
        if is_clipped(input_segment):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # 填充预分配数组
        output_audio[current_pos:current_pos+segment_len] = input_segment  # 修改此行
        current_pos += segment_len  # 必须添加
        remaining_length -= segment_len  # 必须添加
        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio[current_pos:current_pos+silence_len] = silence[:silence_len]
            current_pos += silence_len
            remaining_length -= silence_len
    # 截取实际使用的部分
    output_audio = output_audio[:current_pos]
    
    
    if tries_left == 0 and not is_clean and 'noisedirs' in params.keys():
        print("There are not enough non-clipped files in the " + noisedirs[idx_n_dir] + \
              " directory to complete the audio build")
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length'] * params['fs'])
    if is_clean:
        activity_threshold = params['clean_activity_threshold']
    else:
        activity_threshold = params['noise_activity_threshold']

    while True:
        audio, source_files, new_clipped_files, index = \
            build_audio(is_clean, params, index, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index

'''
# 引入多线程并行化
from concurrent.futures import ThreadPoolExecutor

def parallel_gen_audio(is_clean, params, index, audio_samples_length=-1):
    with ThreadPoolExecutor(max_workers=4) as executor:
        future = executor.submit(gen_audio, is_clean, params, index, audio_samples_length)
        return future.result()
'''

def process_batch(params, file_num_start, file_num_end, queue):
    '''Process a batch of files in a separate process'''
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    # 获取当前进程的索引
    process_idx = multiprocessing.current_process()._identity[0] - 1  # 进程索引从0开始
    # 使用固定种子 + 进程索引的方式
    fixed_seed = 42  # 固定种子
    np.random.seed(fixed_seed + process_idx)
    random.seed(fixed_seed + process_idx)
    
    # 在子进程初始化时创建独立logger
    logger = logging.getLogger(f"Process-{process_idx}")
    logger.setLevel(logging.INFO)
    
    # 创建文件handler时添加进程标识，并保存到log_dir中
    log_dir = params['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(log_dir, f'multiple_file_access_{process_idx}.log'))
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    clean_index = 0
    noise_index = 0
    file_num = file_num_start

    # 将.lst文件保存到log_dir中
    train_list_file = os.path.join(log_dir, f'train_{file_num_start}_{file_num_end}.lst')
    with open(train_list_file, 'w') as lst:
        while file_num <= file_num_end:

            clean, clean_sf, clean_cf, clean_laf, clean_index = \
                gen_audio(True, params, clean_index)
                
            rir_index = random.randint(0, len(params['myrir'])-1)
            my_rir_path = os.path.join('datasets/impulse_responses', params['myrir'][rir_index])
            _, samples_rir = wavfile.read(my_rir_path)

            # 通道选择逻辑
            if samples_rir.ndim == 1:
                samples_rir_ch = samples_rir
            else:
                channel_idx = int(params['mychannel'][rir_index]) - 1
                samples_rir_ch = samples_rir[:, channel_idx]

            # 应用混响
            clean = add_pyreverb(clean, samples_rir_ch)

            noise, noise_sf, noise_cf, noise_laf, noise_index = \
                gen_audio(False, params, noise_index, len(clean))
   

            clean_clipped_files += clean_cf
            clean_low_activity_files += clean_laf
            noise_clipped_files += noise_cf
            noise_low_activity_files += noise_laf

            # mix clean speech and noise
            # if specified, use specified SNR value
            if not params['randomize_snr']:
                snr = params['snr']
            # use a randomly sampled SNR value between the specified bounds
            else:
                snr = np.random.randint(params['snr_lower'], params['snr_upper'])

            # clean_snr, noise_snr, noisy_snr, target_level = snr_mixer(params=params,
            #                                                           clean=clean,
            #                                                           noise=noise,
            #                                                           snr=snr)
            # Uncomment the below lines if you need segmental SNR and comment the above lines using snr_mixer
            clean_snr, noise_snr, noisy_snr, target_level = segmental_snr_mixer(params=params,
                                                                                clean=clean,
                                                                                noise=noise,
                                                                                snr=snr)
            # unexpected clipping
            if is_clipped(clean_snr) or is_clipped(noise_snr) or is_clipped(noisy_snr):
                print("Warning: File #" + str(file_num) + " has unexpected clipping, " + \
                      "returning without writing audio to disk")
                continue

            clean_source_files += clean_sf
            noise_source_files += noise_sf

            cleanfilename = 'clean_fileid_'+str(file_num)+'.wav'
            noisefilename = 'noise_fileid_'+str(file_num)+'.wav'
            noisyfilename = 'noisy_fileid_'+str(file_num)+ '_snr' + \
                            str(snr) + '_tl' + str(target_level) + '.wav'

            noisypath = os.path.join(params['noisyspeech_dir'], noisyfilename)
            cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
            noisepath = os.path.join(params['noise_proc_dir'], noisefilename)
            lst.write(' '.join((noisypath, cleanpath, f'{params["audio_length"]}')) + '\n')

            audio_signals = [noisy_snr, clean_snr, noise_snr]
            file_paths = [noisypath, cleanpath, noisepath]

            file_num += 1
            for i in range(len(audio_signals)):
                try:
                    audiowrite(file_paths[i], audio_signals[i], params['fs'])
                except Exception as e:
                    print(str(e))
                    
    queue.put((clean_source_files, clean_clipped_files, clean_low_activity_files,
               noise_source_files, noise_clipped_files, noise_low_activity_files))

def main_gen(params):
    '''Calls process_batch() in parallel to generate the audio signals'''

    #breakpoint()
    file_num_start = params['fileindex_start']
    file_num_end = params['fileindex_end']
    total_files = file_num_end - file_num_start + 1
    files_per_process = total_files // num_processes
    
    processes = []
    queue = Queue()

    for i in range(num_processes):
        start = file_num_start + i * files_per_process
        end = start + files_per_process - 1 if i < num_processes - 1 else file_num_end
        # 为每个进程创建独立参数
        process_params = params.copy()
        process_params.update({
            'cleanfilenames': params['process_resources']['clean'][i],
            'noisefilenames': params['process_resources']['noise'][i],
            'myrir': params['process_resources']['rir'][i],
            'mychannel': params['process_resources']['rirchannel'][i],
            'myt60': params['process_resources']['rirt60'][i]
        })
        p = Process(target=process_batch, args=(process_params, start, end, queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Collect results from all processes
    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    while not queue.empty():
        result = queue.get()
        clean_source_files += result[0]
        clean_clipped_files += result[1]
        clean_low_activity_files += result[2]
        noise_source_files += result[3]
        noise_clipped_files += result[4]
        noise_low_activity_files += result[5]

    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
           noise_source_files, noise_clipped_files, noise_low_activity_files

def split_resources(res_list, n):
    k, m = divmod(len(res_list), n)
    return [res_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def main_body():
    '''Main body of this file'''
    
    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='noisyspeech_synthesizer.cfg',
                        help='Read noisyspeech_synthesizer.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='noisy_speech')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    # 将日志文件保存到log_dir中
    log_dir = utils.get_dir(cfg, 'log_dir', 'Logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,  # 将级别改为 INFO 过滤 DEBUG 信息
        format='%(asctime)s - %(message)s',  # 简化格式
        filename=os.path.join(log_dir, 'multiple_file_access.log'),
        filemode='w'
    )

    clean_dir = os.path.join(os.path.dirname(__file__), 'datasets/clean')

    if cfg['speech_dir'] != 'None':
        clean_dir = cfg['speech_dir']
    if not os.path.exists(clean_dir):
        assert False, ('Clean speech data is required')

    noise_dir = os.path.join(os.path.dirname(__file__), 'datasets/noise')

    if cfg['noise_dir'] != 'None':
        noise_dir = cfg['noise_dir']
    if not os.path.exists:
        assert False, ('Noise data is required')

    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])

    # clean singing speech
    params['use_singing_data'] = int(cfg['use_singing_data'])
    params['clean_singing'] = str(cfg['clean_singing'])
    params['singing_choice'] = int(cfg['singing_choice'])

    # clean emotional speech
    params['use_emotion_data'] = int(cfg['use_emotion_data'])
    params['clean_emotion'] = str(cfg['clean_emotion'])

    # clean mandarin speech
    params['use_mandarin_data'] = int(cfg['use_mandarin_data'])
    params['clean_mandarin'] = str(cfg['clean_mandarin'])

    # rir
    params['rir_choice'] = int(cfg['rir_choice'])
    params['lower_t60'] = float(cfg['lower_t60'])
    params['upper_t60'] = float(cfg['upper_t60'])
    params['rir_table_csv'] = str(cfg['rir_table_csv'])
    params['clean_speech_t60_csv'] = str(cfg['clean_speech_t60_csv'])

    if cfg['fileindex_start'] != 'None' and cfg['fileindex_end'] != 'None':
        params['num_files'] = int(cfg['fileindex_end']) - int(cfg['fileindex_start'])
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])
    else:
        params['num_files'] = int((params['total_hours'] * 60 * 60) / params['audio_length'])
        params['fileindex_start'] = 0
        params['fileindex_end'] = params['num_files']

    print('Number of files to be synthesized:', params['num_files'])

    params['is_test_set'] = utils.str2bool(cfg['is_test_set'])
    params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
    params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
    params['snr_lower'] = int(cfg['snr_lower'])
    params['snr_upper'] = int(cfg['snr_upper'])

    params['randomize_snr'] = utils.str2bool(cfg['randomize_snr'])
    params['target_level_lower'] = int(cfg['target_level_lower'])
    params['target_level_upper'] = int(cfg['target_level_upper'])

    if 'snr' in cfg.keys():
        params['snr'] = int(cfg['snr'])
    else:
        params['snr'] = int((params['snr_lower'] + params['snr_upper']) / 2)

    params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
    params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
    params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')

    if 'speech_csv' in cfg.keys() and cfg['speech_csv'] != 'None':
        cleanfilenames = pd.read_csv(cfg['speech_csv'])
        cleanfilenames = cleanfilenames['filename']
    else:
        cleanfilenames = []
        for path in Path(clean_dir).rglob('*.wav'):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
    # add singing voice to clean speech
    if params['use_singing_data'] == 1:
        all_singing = []
        for path in Path(params['clean_singing']).rglob('*.wav'):
            all_singing.append(str(path.resolve()))

        if params['singing_choice'] == 1:  # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]

        elif params['singing_choice'] == 2:  # female speakers
            mysinging = [s for s in all_singing if "female" in s]

        elif params['singing_choice'] == 3:  # both male and female
            mysinging = all_singing
        else:  # default both male and female
            mysinging = all_singing

        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles = cleanfilenames + mysinging
    else:
        all_cleanfiles = cleanfilenames

    # add emotion data to clean speech
    if params['use_emotion_data'] == 1:
        all_emotion = []
        for path in Path(params['clean_emotion']).rglob('*.wav'):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else:
        print('NOT using emotion data for training!')

    # add mandarin data to clean speech
    if params['use_mandarin_data'] == 1:
        all_mandarin = []
        for path in Path(params['clean_mandarin']).rglob('*.wav'):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else:
        print('NOT using non-english (Mandarin) data for training!')

    shuffle(all_cleanfiles)

    params['cleanfilenames'] = all_cleanfiles
    params['num_cleanfiles'] = len(params['cleanfilenames'])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if 'noise_csv' in cfg.keys() and cfg['noise_csv'] != 'None':
        noisefilenames = pd.read_csv(cfg['noise_csv'])
        noisefilenames = noisefilenames['filename'].tolist()
    else:
        noisefilenames = []
        for path in Path(noise_dir).rglob('*.wav'):
            noisefilenames.append(str(path.resolve()))
    
    # 预先解析所有文件路径
    params['noisefilenames'] = noisefilenames

    if len(noisefilenames) != 0:
        shuffle(noisefilenames)
        params['noisefilenames'] = noisefilenames
    else:
        noisedirs = glob.glob(os.path.join(noise_dir, '*'))
        if cfg['noise_types_excluded'] != 'None':
            dirstoexclude = cfg['noise_types_excluded'].split(',')
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params['noisedirs'] = noisedirs

    # rir
    temp = pd.read_csv(params['rir_table_csv'], skiprows=[1], sep=',', header=None,
                       names=['wavfile', 'channel', 'T60_WB', 'C50_WB', 'isRealRIR'])
    temp.keys()
    # temp.wavfile

    rir_wav = temp['wavfile'][1:]  # 115413
    rir_channel = temp['channel'][1:]
    rir_t60 = temp['T60_WB'][1:]
    rir_isreal = temp['isRealRIR'][1:]

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    myrir = []
    mychannel = []
    myt60 = []

    lower_t60 = params['lower_t60']
    upper_t60 = params['upper_t60']

    if params['rir_choice'] == 1:  # real 3076 IRs
        real_indices = [i for i, x in enumerate(rir_isreal2) if x == "1"]

        chosen_i = []
        for i in real_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]


    elif params['rir_choice'] == 2:  # synthetic 112337 IRs
        synthetic_indices = [i for i, x in enumerate(rir_isreal2) if x == "0"]

        chosen_i = []
        for i in synthetic_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params['rir_choice'] == 3:  # both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    else:  # default both real and synthetic
        all_indices = [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir = [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    params['myrir'] = myrir
    params['mychannel'] = mychannel
    params['myt60'] = myt60
    
    process_resources = {
        'clean': split_resources(cleanfilenames, num_processes),
        'noise': split_resources(noisefilenames, num_processes),
        'rir': split_resources(myrir, num_processes),
        'rirchannel': split_resources(mychannel, num_processes),
        'rirt60': split_resources(myt60, num_processes)
    }
    params['process_resources'] = process_resources

    # 将log_dir添加到params中，以便在process_batch中使用
    params['log_dir'] = log_dir

    # Call main_gen() to generate audio
    clean_source_files, clean_clipped_files, clean_low_activity_files, \
    noise_source_files, noise_clipped_files, noise_low_activity_files = main_gen(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = utils.get_dir(cfg, 'log_dir', 'Logs')

    utils.write_log_file(log_dir, 'source_files.csv', clean_source_files + noise_source_files)
    utils.write_log_file(log_dir, 'clipped_files.csv', clean_clipped_files + noise_clipped_files)
    utils.write_log_file(log_dir, 'low_activity_files.csv', \
                         clean_low_activity_files + noise_low_activity_files)

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(clean_source_files) + len(clean_clipped_files) + len(clean_low_activity_files)
    total_noise = len(noise_source_files) + len(noise_clipped_files) + len(noise_low_activity_files)
    pct_clean_clipped = round(len(clean_clipped_files) / total_clean * 100, 1)
    pct_noise_clipped = round(len(noise_clipped_files) / total_noise * 100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files) / total_clean * 100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files) / total_noise * 100, 1)

    print("Of the " + str(total_clean) + " clean speech files analyzed, " + \
          str(pct_clean_clipped) + "% had clipping, and " + str(pct_clean_low_activity) + \
          "% had low activity " + "(below " + str(params['clean_activity_threshold'] * 100) + \
          "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) + \
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " + \
          "(below " + str(params['noise_activity_threshold'] * 100) + "% active percentage)")


if __name__ == '__main__':
    print('num_processes:', num_processes)
    main_body()