import torch 

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, image_data):
        self.image_data = image_data

    def __getitem__(self, idx):
        image, _ = self.image_data[idx]
        fake_text = torch.randn(100)
        return {'image': image, 'text': fake_text}, torch.randn(64)

    def __len__(self):
        return len(self.image_data)