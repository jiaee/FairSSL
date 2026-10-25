import os
import torch
import torch.nn as nn
import wandb
import torch.optim as optim
from types import SimpleNamespace
from Utils.metric_calculate import calculate_metrics
from Models.FeatNet import EEGFeatNet, AudioFeatNet, MultiModalCNN
from DataSets.MODMA import prepare_subject_dataloader
import copy
import numpy as np
from Utils.loss_type_dataloader import loss_type_dataloader
import random
from SSL.VICReg import VICReg_M1, VICReg_M234

def _unpack_modma_batch(batch):
    if len(batch) == 6:
        audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = batch
    else:
        audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids = batch
        eeg_ids = audio_ids
    return audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids


def _mean_eeg_by_audio_subject(audio_features, eeg_features, audio_ids, eeg_ids):
    groups = {}
    for idx, sid in enumerate(eeg_ids):
        groups.setdefault(sid, []).append(idx)
    pooled = []
    for sid in audio_ids:
        idxs = groups.get(sid)
        if idxs:
            pooled.append(eeg_features[idxs].mean(dim=0))
        else:
            pooled.append(eeg_features.mean(dim=0))
    if not pooled:
        return eeg_features.new_empty((0, eeg_features.size(1)))
    return torch.stack(pooled, dim=0)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')       

def evaluate_model(test_dataloader, multimodal_model, linear_classifier):
    multimodal_model.eval()
    linear_classifier.eval()
    y_pred = []
    y_true = []
    y_gender = []
    y_prob = []
    y_IDs= []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = _unpack_modma_batch(batch)
            audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)
            do_pooling = getattr(linear_classifier, "do_pooling", True)
            if do_pooling:
                pooled_eeg_features_repeated = _mean_eeg_by_audio_subject(audio_features, eeg_features, audio_ids, eeg_ids)
                combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)
            else:
                combined_features = torch.cat((audio_features, eeg_features), dim=1)


            # unique_subj = sorted(set(IDs_tensor))           # e.g. ['P001', 'P005', 'P034']
            # SUBJ2IDX    = {sid: i for i, sid in enumerate(unique_subj)}

            # subj_ids = torch.tensor(
            #         [SUBJ2IDX[s] for s in IDs_tensor], 
            #         device=eeg_features.device
            # )

            # unique_ids, inverse = torch.unique(subj_ids, return_inverse=True)
            # subj_means = torch.zeros(
            #         len(unique_ids), eeg_features.size(1),
            #         device=eeg_features.device
            # ).index_add_(0, inverse, eeg_features)        # sum over rows
            # counts = torch.bincount(inverse).float().unsqueeze(1)
            # subj_means = subj_means / counts              # → mean per subject

            # pooled_eeg_features_repeated = subj_means[inverse]
            # combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)

            # Use the linear classifier for final classification
            output = linear_classifier(combined_features)
            predicted = (output.squeeze() > 0).long()
            # Append results for metrics calculation later
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(label_batch.cpu().numpy())
            y_gender.extend(gender_tensor.cpu().numpy())
            y_IDs.extend(audio_ids)
            y_prob.extend(output.cpu().numpy())
    return y_pred, y_true, y_gender, y_prob, y_IDs


def setup_linear_classifier(audio_features_dim, eeg_features_dim, device, config):
    input_dim = audio_features_dim + eeg_features_dim
    linear_classifier = nn.Linear(input_dim, 1).to(device)
    lr = getattr(config, "linear_lr", None)
    if lr is None:
        lr = getattr(config, "learning_rate")
    linear_optimizer = optim.Adam(linear_classifier.parameters(), lr=lr)

    linear_classifier.do_pooling = True

    return linear_classifier, linear_optimizer

    

def linear_probing(multimodal_model, train_dataloader, val_dataloader, setup_linear_classifier, device, MODEL_SAVE_DIR, config):

    # multimodal_model.eval()
    # for param in multimodal_model.parameters():
    #     param.requires_grad = False

    best_val_accuracy = 0
    patience_counter = 0
    best_multimodal_model = None
    best_linear_classifier = None

    audio_features_dim = None
    eeg_features_dim = None

    # Extract feature dimensions
    for batch_idx, batch in enumerate(train_dataloader):
        audio_batch, eeg_batch, _, _, _, _ = _unpack_modma_batch(batch)
        with torch.no_grad():
            audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)
            audio_features_dim = audio_features.shape[1]
            eeg_features_dim = eeg_features.shape[-1]
        break

    linear_classifier, linear_optimizer = setup_linear_classifier(audio_features_dim, eeg_features_dim, device, config)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # Train linear classifier for 5 epochs
    for probe_epoch in range(getattr(config, "probe_epochs", 5)):
        linear_classifier.train()
        total_probe_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = _unpack_modma_batch(batch)

            with torch.no_grad():
                audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)
                do_pooling = getattr(linear_classifier, "do_pooling", True)
                if do_pooling:
                    pooled_eeg_features_repeated = _mean_eeg_by_audio_subject(audio_features, eeg_features, audio_ids, eeg_ids)
                    combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)
                else:
                    combined_features = torch.cat((audio_features, eeg_features), dim=1)

                # unique_subj = sorted(set(IDs_tensor)) 
                # SUBJ2IDX    = {sid: i for i, sid in enumerate(unique_subj)}

                # subj_ids = torch.tensor(
                #         [SUBJ2IDX[s] for s in IDs_tensor], 
                #         device=eeg_features.device
                # )

                # unique_ids, inverse = torch.unique(subj_ids, return_inverse=True)
                # subj_means = torch.zeros(
                #         len(unique_ids), eeg_features.size(1),
                #         device=eeg_features.device
                # ).index_add_(0, inverse, eeg_features) 
                # counts = torch.bincount(inverse).float().unsqueeze(1)
                # subj_means = subj_means / counts

                # pooled_eeg_features_repeated = subj_means[inverse]
                # combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)

            logits = linear_classifier(combined_features)
            loss = criterion(logits.squeeze(), label_batch.float())

            linear_optimizer.zero_grad()
            loss.backward()
            linear_optimizer.step()

            total_probe_loss += loss.item()

        # Validation after linear probing
        linear_classifier.eval()
        y_true = []
        y_pred = []
        y_gender = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = _unpack_modma_batch(batch)
                audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)

                do_pooling = getattr(linear_classifier, "do_pooling", True)
                if do_pooling:
                    pooled_eeg_features_repeated = _mean_eeg_by_audio_subject(audio_features, eeg_features, audio_ids, eeg_ids)
                    combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)
                else:
                    combined_features = torch.cat((audio_features, eeg_features), dim=1)


                # unique_subj = sorted(set(IDs_tensor))
                # SUBJ2IDX    = {sid: i for i, sid in enumerate(unique_subj)}

                # subj_ids = torch.tensor(
                #         [SUBJ2IDX[s] for s in IDs_tensor],   # or any numeric encoding
                #         device=eeg_features.device
                # )

                # unique_ids, inverse = torch.unique(subj_ids, return_inverse=True)
                # subj_means = torch.zeros(
                #         len(unique_ids), eeg_features.size(1),
                #         device=eeg_features.device
                # ).index_add_(0, inverse, eeg_features)        # sum over rows
                # counts = torch.bincount(inverse).float().unsqueeze(1)
                # subj_means = subj_means / counts

                # pooled_eeg_features_repeated = subj_means[inverse] 
                # combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)

                outputs = linear_classifier(combined_features)
                predicted = (outputs.squeeze() > 0).long()

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(label_batch.cpu().numpy())
                y_gender.extend(gender_tensor.cpu().numpy())

        # Calculate validation metrics
        SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar, _ = calculate_metrics(
            y_pred=y_pred,
            y_labels=y_true,
            gender_list=y_gender
        )

        print(f"Epoch [{probe_epoch + 1}/5], "
              f"Linear Probing (Val) Accuracy (over 3 mini-epochs): {Acc:.2f}%")

        wandb.log({
            "linear_probe_epoch": probe_epoch + 1,
            "linear_probe_loss": total_probe_loss / len(train_dataloader),
            "val_accuracy_probe_epoch": Acc})

        if Acc > best_val_accuracy:
            best_val_accuracy = Acc
            patience_counter = 0

            best_multimodal_model = multimodal_model
            best_linear_classifier = linear_classifier

            if MODEL_SAVE_DIR is not None:
                os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
                multimodal_model_path = os.path.join(MODEL_SAVE_DIR, "best_multimodal_model.pth")
                linear_classifier_path = os.path.join(MODEL_SAVE_DIR, "best_linear_classifier.pth")

                if isinstance(multimodal_model, nn.DataParallel):
                    torch.save(multimodal_model.module.state_dict(), multimodal_model_path)
                else:
                    torch.save(multimodal_model.state_dict(), multimodal_model_path)

                torch.save(linear_classifier.state_dict(), linear_classifier_path)
                print(f"New best val_accuracy={best_val_accuracy:.2f}%. Models saved.")
            wandb.log({"val_accuracy": best_val_accuracy})

    return best_val_accuracy, (best_multimodal_model, best_linear_classifier)

def linear_probing_test(multimodal_model, train_dataloader, setup_linear_classifier, device, MODEL_SAVE_DIR, config):

    # multimodal_model.eval()
    # for param in multimodal_model.parameters():
    #     param.requires_grad = False
    audio_features_dim = None
    eeg_features_dim = None

    # Extract feature dimensions
    for batch_idx, batch in enumerate(train_dataloader):
        audio_batch, eeg_batch, _, _, _, _ = _unpack_modma_batch(batch)
        with torch.no_grad():
            audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)
            audio_features_dim = audio_features.shape[1]
            eeg_features_dim = eeg_features.shape[-1]
        break

    linear_classifier, linear_optimizer = setup_linear_classifier(audio_features_dim, eeg_features_dim, device, config)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # Train linear classifier for suitable epochs based on the tuning results
    for probe_epoch in range(5):
        linear_classifier.train()
        total_probe_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = _unpack_modma_batch(batch)

            with torch.no_grad():
                audio_features, eeg_features = multimodal_model(audio_batch, eeg_batch)
                do_pooling = getattr(linear_classifier, "do_pooling", True)
                if do_pooling:
                    pooled_eeg_features_repeated = _mean_eeg_by_audio_subject(audio_features, eeg_features, audio_ids, eeg_ids)
                    combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)
                else:
                    combined_features = torch.cat((audio_features, eeg_features), dim=1)


                # unique_subj = sorted(set(IDs_tensor))
                # SUBJ2IDX    = {sid: i for i, sid in enumerate(unique_subj)}

                # subj_ids = torch.tensor(
                #         [SUBJ2IDX[s] for s in IDs_tensor],
                #         device=eeg_features.device
                # )

                # unique_ids, inverse = torch.unique(subj_ids, return_inverse=True)
                # subj_means = torch.zeros(
                #         len(unique_ids), eeg_features.size(1),
                #         device=eeg_features.device
                # ).index_add_(0, inverse, eeg_features)
                # counts = torch.bincount(inverse).float().unsqueeze(1)
                # subj_means = subj_means / counts

                # pooled_eeg_features_repeated = subj_means[inverse]
                # combined_features = torch.cat((audio_features, pooled_eeg_features_repeated), dim=1)

            logits = linear_classifier(combined_features)
            loss = criterion(logits.squeeze(), label_batch.float())

            linear_optimizer.zero_grad()
            loss.backward()
            linear_optimizer.step()

            total_probe_loss += loss.item()

    return linear_classifier

def train_final_model(config, train_data_by_subject, test_data_by_subject, seed):
    from Utils.loss_type_dataloader import loss_type_dataloader

    def _as_sid_list(x):
        # Make subject ids usable as dict keys inside VICReg (and other losses).
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        # list/tuple/iterable
        return list(x)

    # --- seed ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, _ = loss_type_dataloader(
        config["loss_type"],
        train_data_by_subject,
        test_data_by_subject,
        config["subjects_per_batch"],
        epoch=0,
        device=device
    )
    _, test_dataloader = loss_type_dataloader(
        config["loss_type"],
        train_data_by_subject,
        test_data_by_subject,
        config["subjects_per_batch"],
        epoch=0,
        device=device
    )

    cnn_audio_encoder = AudioFeatNet(input_shape=(1024, 128)).to(device)
    cnn_eeg_encoder   = EEGFeatNet(input_shape=(128, 500)).to(device)
    initialize_weights(cnn_audio_encoder)
    initialize_weights(cnn_eeg_encoder)

    multimodal_model = MultiModalCNN(cnn_audio_encoder, cnn_eeg_encoder).to(device)
    if torch.cuda.device_count() > 1:
        multimodal_model = nn.DataParallel(multimodal_model)

    with torch.no_grad():
        dummy_audio = torch.zeros(1, 1, 1024, 128, device=device)
        dummy_eeg   = torch.zeros(1, 1, 128, 500, device=device)
        dim_mod1 = cnn_audio_encoder(dummy_audio).shape[1]
        dim_mod2 = cnn_eeg_encoder(dummy_eeg).shape[1]

    family  = str(config["loss_family"]).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(config["loss_type"]).upper()

    if family != "vicreg":
        raise ValueError(f"Unsupported FairWell loss_family: {family}")
    if variant == "M1":
        loss_fn = VICReg_M1(SimpleNamespace(**config))
    elif variant in {"M2", "M3", "M4"}:
        loss_fn = VICReg_M234(dim_mod1, dim_mod2, SimpleNamespace(**config))
    else:
        raise ValueError(f"Unsupported FairWell loss_type: {variant}")

    loss_fn = loss_fn.to(device)
    with torch.no_grad():
        B0 = 4
        sid0 = list(range(B0))
        _ = loss_fn(torch.randn(B0, dim_mod1, device=device),
                    torch.randn(B0, dim_mod2, device=device),
                    sid0, sid0)

    optimizer = optim.Adam(
        list(multimodal_model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"])
    )

    num_epochs = int(config["epoch"])
    for epoch in range(num_epochs):
        train_dataloader, _ = loss_type_dataloader(
            config["loss_type"],
            train_data_by_subject,
            test_data_by_subject,
            config["subjects_per_batch"],
            epoch=epoch,
            device=device
        )

        avg_loss = train_one_epoch(
            multimodal_model,
            train_dataloader,
            optimizer,
            loss_fn,
            family,
            variant,
            device
        )

    probe_train_loader = prepare_subject_dataloader(
        train_data_by_subject,
        config["subjects_per_batch"],
        device,
        single_label=False
    )

    _, (multimodal_model, linear_classifier) = linear_probing(
        multimodal_model=multimodal_model,
        train_dataloader=probe_train_loader,
        val_dataloader=test_dataloader,
        setup_linear_classifier=setup_linear_classifier,
        device=device,
        MODEL_SAVE_DIR=None,
        config=SimpleNamespace(**config)
    )

    y_pred, y_true, y_gender, y_logits, y_ids = evaluate_model(
        test_dataloader,
        multimodal_model,
        linear_classifier
    )

    return y_pred, y_true, y_gender, y_logits, y_ids



def train_one_epoch(multimodal_model, train_dataloader, optimizer, loss_fn, loss_family, loss_type, device):
    multimodal_model.train()
    loss_fn.train()

    family  = str(loss_family).lower()
    if family == "fairwell":
        family = "vicreg"
    variant = str(loss_type).upper()
    if family != "vicreg" or variant not in {"M1", "M2", "M3", "M4"}:
        raise ValueError(f"Unsupported FairWell loss: {loss_family}/{loss_type}")

    def _as_sid_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return list(x)

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(train_dataloader):
        audio_batch, eeg_batch, label_batch, gender_tensor, audio_ids, eeg_ids = _unpack_modma_batch(batch)
        audio_batch = audio_batch.to(device).float()
        eeg_batch   = eeg_batch.to(device).float()
        audio_sid = _as_sid_list(audio_ids)
        eeg_sid = _as_sid_list(eeg_ids)

        optimizer.zero_grad()

        z_m1, z_m2 = multimodal_model(audio_batch, eeg_batch)

        loss = loss_fn(z_m1, z_m2, audio_sid, eeg_sid)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)

