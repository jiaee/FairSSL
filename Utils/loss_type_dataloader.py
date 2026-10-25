from DataSets.MODMA import prepare_subject_dataloader

def loss_type_dataloader(loss_type, train_data_by_subject, val_data_by_subject, subjects_per_batch, epoch, device):
    loss_type = str(loss_type).upper()
    if loss_type in {'M1', 'M2'}:
        train_dataloader_ml = prepare_subject_dataloader(train_data_by_subject, subjects_per_batch, device, single_label=False)
        cur_loader = train_dataloader_ml
    elif loss_type == 'M3': 
        train_dataloader_sl = prepare_subject_dataloader(train_data_by_subject, subjects_per_batch, device, single_label=True )
        cur_loader = train_dataloader_sl
    elif loss_type == 'M4':
        train_dataloader_ml = prepare_subject_dataloader(train_data_by_subject, subjects_per_batch, device, single_label=False)
        train_dataloader_sl = prepare_subject_dataloader(train_data_by_subject, subjects_per_batch, device, single_label=True )
        cur_loader = train_dataloader_sl if (epoch + 1) % 2 == 1 else train_dataloader_ml
    else:
        raise ValueError(f"Unsupported FairWell loss_type: {loss_type}")
    
    val_dataloader  = prepare_subject_dataloader(val_data_by_subject, subjects_per_batch, device, single_label=False)
    return cur_loader, val_dataloader
