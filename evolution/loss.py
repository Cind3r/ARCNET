import torch.nn.functional as F
import torch

def compute_loss(model, X, y, population=None, lambda_entropy=0.01, lambda_manifold=0.01, lambda_message=0.01):
    """
    Computes a composite loss for ARCNET architecture.
    - Standard prediction loss (cross-entropy or MSE)
    - Entropy regularization (encourages output diversity)
    - Manifold regularization (optional, encourages smoothness in latent space)
    - Message passing consistency (optional, encourages agreement with neighbors)

    For pure classification: set lambda_entropy=0.01, others to 0. For manifold/message-passing 
    regularization: set lambda_manifold and/or lambda_message to small positive values. Pass 
    population if you want manifold regularization.
    Args:
        model: The ARCNET model to evaluate.
        X: Input data tensor.
        y: Target labels or values tensor.
        population: Optional list of other models for manifold regularization.
        lambda_entropy: Weight for entropy regularization (default 0.01
        lambda_manifold: Weight for manifold regularization (default 0.01)
        lambda_message: Weight for message passing consistency (default 0.01)
    Returns:
        total_loss: Computed loss value.
    """
    
    model.eval()
    with torch.no_grad():
        output = model(X)
        # 1. Prediction loss (classification or regression)
        if output.shape == y.shape or y.dtype == torch.float32:
            # Regression or autoencoder
            pred_loss = F.mse_loss(output, y)
        else:
            # Classification
            if y.ndim == 2 and y.shape[1] > 1:
                y = y.argmax(dim=1)
            pred_loss = F.cross_entropy(output, y)

        # 2. Entropy regularization (encourage diverse predictions)
        probs = F.softmax(output, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        entropy_loss = -lambda_entropy * entropy  # maximize entropy

        # 3. Manifold regularization (optional, if model has .position attribute)
        manifold_loss = 0.0
        if hasattr(model, "position") and population is not None:
            # Encourage smoothness: penalize large distances to neighbors
            neighbors = [m for m in population if m is not model]
            if neighbors:
                dists = [torch.norm(model.position - n.position) for n in neighbors]
                manifold_loss = lambda_manifold * torch.stack(dists).mean()

        # 4. Message passing consistency (optional, if model has .process_messages)
        message_loss = 0.0
        if hasattr(model, "process_messages"):
            try:
                msg = model.process_messages()
                if msg is not None and hasattr(model, "last_hidden") and model.last_hidden is not None:
                    message_loss = lambda_message * F.mse_loss(msg, model.last_hidden)
            except Exception:
                pass

        total_loss = pred_loss + entropy_loss + manifold_loss + message_loss
        return total_loss.item()


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

