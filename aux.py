from preresnet import PreResNet20
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from temperature_scaling.temperature_scaling import ModelWithTemperature
from copy import deepcopy
from tqdm.auto import tqdm
from swa_gaussian.swag.posteriors import SWAG
from swa_gaussian.swag.utils import bn_update
import os

############################################################
###################  Visualization  ########################
############################################################


def plot_histogram_eq_width(ax, data1: list | np.ndarray, data2: list | np.ndarray, title: str) -> float:
    # Define the common bins
    bins = np.linspace(min(min(data1), min(data2)),
                       max(max(data1), max(data2)), num=20)

    # Plot histograms
    ax.hist(data1, bins=bins, alpha=0.5, label='InD')
    ax.hist(data2, bins=bins, alpha=0.5, label='OOD')

    labels = np.hstack([np.zeros(len(data1)), np.ones(len(data2))])
    scores = np.hstack([data1, data2])

    roc_auc = round(roc_auc_score(labels, scores), 3)

    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend([f'InD', f'OOD', f'ROC AUC: {roc_auc}'], loc='upper right')

    return roc_auc


def plot_images(dataset: torch.utils.data.Dataset, indices: np.ndarray) -> None:
    fig = plt.figure(figsize=(len(indices)*3, 3))
    # to_pil = transforms.ToPILImage()

    for i, idx in enumerate(indices):
        img = dataset.data[idx]
        if img.shape[0] == 3:
            img = img.transpose((1, 2, 0))
        ax = fig.add_subplot(1, len(indices), i+1)
        ax.axis('off')
        ax.imshow(img)

    plt.show()



############################################################
############################################################




############################################################
#################  Probs Collection  #######################
############################################################

def collect_labels_and_probs(model_, dataloader, device):
    model_.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model_(images)
            probs = softmax(outputs, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.hstack(all_labels), np.vstack(all_probs)


def collect_labels_and_probs_ensemble(ensemble_, dataloader, device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            probs_batch = []
            for model_ in ensemble_:
                model_.eval()
                outputs = model_(images)
                probabilities = F.softmax(outputs, dim=-1).cpu().numpy()
                probs_batch.append(probabilities[None])
            all_probs.append(np.mean(np.vstack(probs_batch), axis=0))
            all_labels.append(labels.cpu().numpy())
    return np.hstack(all_labels), np.vstack(all_probs)


def collect_labels_and_probs_swag(swag_model_, dataloader, trainloader, n_samples, device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            prob_samples = []
            for _ in range(n_samples):
                swag_model_.sample()
                bn_update(trainloader, swag_model_)
                swag_model_.eval()
                outputs = swag_model_(images)
                probs = softmax(outputs, dim=1)
                prob_samples.append(probs.cpu().numpy()[None])

            all_labels.append(labels.cpu().numpy())
            all_probs.append(np.mean(np.vstack(prob_samples), axis=0))
    return np.hstack(all_labels), np.vstack(all_probs)

def set_dropout_layers_to_train(model):
    model.eval()
    for child in model.children():
        if isinstance(child, torch.nn.Dropout):
            child.train()
        elif isinstance(child, torch.nn.Module):
            set_dropout_layers_to_train(child)



def collect_labels_and_probs_mcdropout(mc_model, dataloader, n_samples, device):
    set_dropout_layers_to_train(model=mc_model)
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            prob_samples = []
            for _ in range(n_samples):
                outputs = mc_model(images)
                probs = softmax(outputs, dim=1)
                prob_samples.append(probs.cpu().numpy()[None])

            all_labels.append(labels.cpu().numpy())
            all_probs.append(np.mean(np.vstack(prob_samples), axis=0))
    return np.hstack(all_labels), np.vstack(all_probs)

############################################################
############################################################





############################################################
#################  Scores computation  #####################
############################################################

def compute_scores_ensemble(ensemble, data_loader, device):
    mean_entropy_scores = []
    entropy_mean_scores = []
    maxprob_scores = []
    with torch.no_grad():
        for data in data_loader:
            images, _ = data
            images = images.to(device)

            single_model_probabilities = []
            for model_ in ensemble:
                model_.eval()
                outputs = model_(images)
                probabilities = F.softmax(outputs, dim=-1).cpu().numpy()
                single_model_probabilities.append(probabilities[None])

            entropy_mean_scores.extend(entropy(
                np.mean(np.vstack(single_model_probabilities), axis=0), axis=-1).tolist())
            mean_entropy_scores.extend(np.mean(
                entropy(np.vstack(single_model_probabilities), axis=-1), axis=0).tolist())
            maxprob_scores.extend(
                (1 - np.max(np.mean(np.vstack(single_model_probabilities), axis=0), axis=1)).tolist())

    return mean_entropy_scores, entropy_mean_scores, maxprob_scores





def compute_scores_mcdropout(mc_model, data_loader, n_samples, device):
    mc_model.train()
    mean_entropy_scores = []
    entropy_mean_scores = []
    maxprob_scores = []
    bald = []
    with torch.no_grad():
        for data in data_loader:
            images, _ = data
            images = images.to(device)

            single_model_probabilities = []
            for _ in range(n_samples):
                outputs = mc_model(images)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                single_model_probabilities.append(probabilities[None])

            entropy_mean_scores.extend(entropy(
                np.mean(np.vstack(single_model_probabilities), axis=0), axis=1).tolist())
            mean_entropy_scores.extend(np.mean(
                entropy(np.vstack(single_model_probabilities), axis=-1), axis=0).tolist())
            maxprob_scores.extend(
                (1 - np.max(np.mean(np.vstack(single_model_probabilities), axis=0), axis=1)).tolist())

    return mean_entropy_scores, entropy_mean_scores, maxprob_scores


def compute_scores_swag(swag_model_, data_loader, trainloader, n_samples, device):
    mean_entropy_scores = []
    entropy_mean_scores = []
    maxprob_scores = []
    bald = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            images, _ = data
            images = images.to(device)

            single_model_probabilities = []
            for _ in range(n_samples):
                swag_model_.sample()
                bn_update(trainloader, swag_model_)
                swag_model_.eval()
                outputs = swag_model_(images)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                single_model_probabilities.append(probabilities[None])

            entropy_mean_scores.extend(entropy(
                np.mean(np.vstack(single_model_probabilities), axis=0), axis=1).tolist())
            mean_entropy_scores.extend(np.mean(
                entropy(np.vstack(single_model_probabilities), axis=-1), axis=0).tolist())
            maxprob_scores.extend(
                (1 - np.max(np.mean(np.vstack(single_model_probabilities), axis=0), axis=1)).tolist())

    return mean_entropy_scores, entropy_mean_scores, maxprob_scores


############################################################
############################################################



############################################################
###############  Deterministic methods  ####################
############################################################



def get_embeddings(model, dataloader, device):
    # Initialize empty lists to hold the outputs and labels
    outputs_list = []
    labels_list = []

    # Switch model to evaluation mode
    model.eval()
    
    # Do not compute gradients
    with torch.no_grad():
        # Iterate over the dataloader
        for inputs, labels in dataloader:
            # Move inputs and labels to the correct device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Run the model on the inputs
            outputs = model(inputs)

            # Add the outputs and labels to our lists
            outputs_list.append(outputs)
            labels_list.append(labels)

    # Concatenate all the outputs and labels tensors together
    outputs_tensor = torch.cat(outputs_list)
    labels_tensor = torch.cat(labels_list)

    return outputs_tensor, labels_tensor


def compute_classwise_mean_cov(embeddings, labels):
    unique_labels = torch.unique(labels)
    means = []
    covs = []

    for lbl in unique_labels:
        lbl_embeddings = embeddings[labels == lbl]

        # Compute the mean
        mean = torch.mean(lbl_embeddings, dim=0)
        means.append(mean)

        # Compute the covariance
        cov = torch.std(lbl_embeddings, dim=0)
        covs.append(cov)

    return means, covs


def log_density_mixture(X, means, covs):
    # Number of mixture components
    M = len(means)
    
    # Number of data points
    N = X.shape[0]

    # Make sure X, means, and covs are all tensors
    X = torch.tensor(X) if not isinstance(X, torch.Tensor) else X
    means = [torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in means]
    covs = [torch.tensor(c) if not isinstance(c, torch.Tensor) else c for c in covs]

    # Initialize log densities
    log_densities = torch.zeros(N, M)

    for i in range(M):
        # Create diagonal covariance matrix from variance
        cov_matrix = torch.diag(covs[i])

        # Create distribution
        distribution = torch.distributions.MultivariateNormal(means[i], cov_matrix)

        # Calculate the log density for each data point
        log_densities[:, i] = distribution.log_prob(X)

    # Combine using log-sum-exp for numerical stability
    max_log_density = torch.max(log_densities, dim=1).values
    log_density = max_log_density + torch.log(torch.sum(torch.exp(log_densities - max_log_density.unsqueeze(-1)), dim=1))

    return log_density


def compute_log_densities(model, dataloader, means, covs, device):
    model.eval()
    log_densities = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack the batch and move tensors to the appropriate device
            inputs = batch[0].to(device)
            # labels = batch[1].to(device)

            # Compute model outputs (embeddings)
            outputs = model(inputs)

            # Compute log densities
            log_density_batch = log_density_mixture(outputs, means, covs)

            # Append to list of log densities
            log_densities.append(log_density_batch)

    # Concatenate all results into a single tensor
    log_densities = torch.cat(log_densities, dim=0)

    return log_densities

############################################################
############################################################



############################################################
###################  Auxilary utils  #######################
############################################################


def load_emsembles(path: str = './helpers/ensembles/', device: str = 'cpu'):
    models = []
    for filename in os.listdir(path):
        if filename.endswith('.pth'):
            model_conf_ = PreResNet20()
            model_ = model_conf_.base(
                *model_conf_.args, num_classes=100, **model_conf_.kwargs)
            model_.load_state_dict(torch.load(os.path.join(path, filename)))
            model_ = model_.to(device)
            model_.eval()
            models.append(model_)
    return models
    

def ensemble_calibration(ensemble, calloader):
    calibrated_ensemble = []
    for model in ensemble:
        calibrated_model = ModelWithTemperature(deepcopy(model))
        calibrated_model.eval()
        calibrated_model.set_temperature(calloader)
        calibrated_ensemble.append(deepcopy(calibrated_model))
    return calibrated_ensemble

############################################################
############################################################