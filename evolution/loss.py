import torch.nn.functional as F

def intrinsic_loss(output, reconstruction, entropy_weight=0.01):
    recon_loss = F.mse_loss(reconstruction, output)
    entropy = -torch.mean(output * torch.log(output + 1e-8))
    return recon_loss - entropy_weight * entropy

def multimodal_selfsupervised_loss(z1, z2, recon, target):
    # contrastive term
    sim = F.cosine_similarity(z1, z2, dim=-1)
    contrastive_loss = 1 - sim.mean()
    
    # reconstruction
    recon_loss = F.mse_loss(recon, target)
    return contrastive_loss + 0.5 * recon_loss
