import os
import random
import numpy as np
import torch

def load_folder_data(folder, num_segments=None):
    if not os.path.exists(folder):
        print(f"Skipping missing folder: {folder}")
        return []
    folder_files = sorted(os.listdir(folder))
    data = []
    for f in folder_files:
        file_path = os.path.join(folder, f)
        if os.path.isfile(file_path):
            try:
                data.append(np.load(file_path))
            except Exception as e:
                print(f"Cannot read file: {file_path}. Error: {e}")
        else:
            print(f"Skipping non-file: {file_path}")
    if num_segments is not None and int(num_segments) > 0:
        num_segments = int(num_segments)
        data = data[:num_segments]
    return data

def load_reshaped_data(subject_dir_eeg, subject_dir_audio,  num_per_aug, eeg_aug_list, audio_aug_list, num_eeg_segments_per_patient, num_audio_segments_per_patient):
    eeg_folders = [
        os.path.join(subject_dir_eeg, 'EEG_10sec'),
    ]

    audio_folders = [
        os.path.join(subject_dir_audio, 'reshaped_audio'),
    ]
    
    aug_eeg_folders = []
    aug_audio_folders = []

    for aug in eeg_aug_list:
        aug_eeg_folders.append(os.path.join(subject_dir_eeg, aug))
        
    for aug in audio_aug_list:
        aug_audio_folders.append(os.path.join(subject_dir_audio, aug))

    eeg_data = []
    audio_data = []
    print(eeg_folders)
    print(audio_folders)

    eeg_data.extend(load_folder_data(os.path.join(subject_dir_eeg, 'EEG_10sec'), num_eeg_segments_per_patient))
    audio_data.extend(load_folder_data(os.path.join(subject_dir_audio, 'reshaped_audio'), num_audio_segments_per_patient))

    for folder in aug_eeg_folders:
        eeg_data.extend(load_folder_data(folder, num_per_aug))

    for folder in aug_audio_folders:
        audio_data.extend(load_folder_data(folder, num_per_aug))

    eeg_data = np.array(eeg_data)
    audio_data = np.array(audio_data)

    return audio_data, eeg_data


def gather_data_from_fold(fold_dir_eeg, fold_dir_audio, num_per_aug, eeg_aug_list, audio_aug_list, num_eeg_segments_per_patient, num_audio_segments_per_patient):
    subject_data = {}

    for condition in ['MDD', 'HC']:
        condition_label = 1 if condition == 'MDD' else 0

        for gender in ['Male', 'Female']:
            gender_label = 1 if gender == 'Male' else 0

            gender_dir_eeg = os.path.join(fold_dir_eeg, condition, gender)
            gender_dir_audio = os.path.join(fold_dir_audio, condition, gender)
            if not os.path.exists(gender_dir_eeg):
                continue
            if not os.path.exists(gender_dir_audio):
                continue

            for subject in os.listdir(gender_dir_eeg):
                subject_dir_eeg = os.path.join(gender_dir_eeg, subject)
                subject_dir_audio = os.path.join(gender_dir_audio, subject)
                audio_data, eeg_data = load_reshaped_data(subject_dir_eeg, 
                                                          subject_dir_audio, 
                                                          num_per_aug, 
                                                          eeg_aug_list, 
                                                          audio_aug_list, 
                                                          num_eeg_segments_per_patient, 
                                                          num_audio_segments_per_patient)

                if len(audio_data) == 0 or len(eeg_data) == 0:
                    continue

                subject_data[subject] = {
                    "audio": list(audio_data),
                    "eeg": list(eeg_data),
                    "labels": [condition_label],
                    "genders": [gender_label],
                    "subjects": [subject],
                }

    return subject_data

def prepare_folds_and_test_data(base_dir_eeg,
                                base_dir_audio, 
                                num_per_aug=None,
                                eeg_aug_list=None, 
                                audio_aug_list=None,
                                eegpatient=None,
                                audiopatient=None):
    eeg_aug_list = eeg_aug_list or []
    audio_aug_list = audio_aug_list or []
    data_dict = {}

    for i in [1, 2, 3, 4, 5]:
        fold_name = f'fold{i}'
        fold_dir_eeg = os.path.join(base_dir_eeg, fold_name)
        fold_dir_audio = os.path.join(base_dir_audio, fold_name)

        fold_data = gather_data_from_fold(
            fold_dir_eeg, 
            fold_dir_audio, 
            num_per_aug, 
            eeg_aug_list, 
            audio_aug_list, 
            eegpatient, 
            audiopatient
        )
        data_dict[fold_name] = fold_data

    test_dir_eeg = os.path.join(base_dir_eeg, 'test')
    test_dir_audio = os.path.join(base_dir_audio, 'test')
    test_data = gather_data_from_fold(
        test_dir_eeg, 
        test_dir_audio, 
        num_per_aug, 
        eeg_aug_list, 
        audio_aug_list, 
        eegpatient, 
        audiopatient
    )
    data_dict['test'] = test_data

    return data_dict


def organize_data_by_subject(audio_data, eeg_data, subjects, labels, genders):
    if isinstance(audio_data, dict):
        return audio_data

    subject_data = {}
    for i, subject in enumerate(subjects):
        if subject not in subject_data:
            subject_data[subject] = {
                "audio": [],
                "eeg": [],
                "labels": [],
                "genders": [],
                "subjects": []
            }
        subject_data[subject]["audio"].append(audio_data[i])
        subject_data[subject]["eeg"].append(eeg_data[i])
        subject_data[subject]["labels"].append(labels[i])
        subject_data[subject]["genders"].append(genders[i])
        subject_data[subject]["subjects"].append(subjects[i])
    return subject_data

def prepare_subject_dataloader(data_by_subject,
                               subjects_per_batch,
                               device,
                               single_label=True):
    if single_label:
        # group by label -> homogeneous batches
        label_groups = {}
        for subj, d in data_by_subject.items():
            lbl = d["labels"][0]
            label_groups.setdefault(lbl, []).append(subj)

        subject_batches = []
        for subj_list in label_groups.values():
            random.shuffle(subj_list)
            for i in range(0, len(subj_list), subjects_per_batch):
                subject_batches.append(subj_list[i:i+subjects_per_batch])

    else:
        # ignore labels -> mixed batches
        all_subjects = list(data_by_subject.keys())
        random.shuffle(all_subjects)
        subject_batches = [
            all_subjects[i : i+subjects_per_batch]
            for i in range(0, len(all_subjects), subjects_per_batch)
        ]

    # turn each batch of subject IDs into tensors
    dataloaders = []
    for batch_subjs in subject_batches:
        batch_audio, batch_eeg = [], []
        batch_lbls, batch_genders = [], []
        batch_audio_IDs, batch_eeg_IDs = [], []
        for subj in batch_subjs:
            audio_segments = data_by_subject[subj]["audio"]
            eeg_segments = data_by_subject[subj]["eeg"]
            label = data_by_subject[subj]["labels"][0]
            gender = data_by_subject[subj]["genders"][0]

            batch_audio.extend(audio_segments)
            batch_eeg.extend(eeg_segments)
            batch_lbls.extend([label] * len(audio_segments))
            batch_genders.extend([gender] * len(audio_segments))
            batch_audio_IDs.extend([subj] * len(audio_segments))
            batch_eeg_IDs.extend([subj] * len(eeg_segments))

        if len(batch_audio) == 0 or len(batch_eeg) == 0:
            continue

        audio_tensor  = torch.tensor(batch_audio,  dtype=torch.float32, device=device)
        eeg_tensor    = torch.tensor(batch_eeg,    dtype=torch.float32, device=device)
        label_tensor  = torch.tensor(batch_lbls,   dtype=torch.long,    device=device)
        gender_tensor = torch.tensor(batch_genders,dtype=torch.long,    device=device)

        dataloaders.append((audio_tensor,
                            eeg_tensor,
                            label_tensor,
                            gender_tensor,
                            batch_audio_IDs,
                            batch_eeg_IDs))
    return dataloaders
