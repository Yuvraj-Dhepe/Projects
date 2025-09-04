import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""### Library Imports""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import random
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import MNIST
    from torch import optim
    return (
        DataLoader,
        Dataset,
        F,
        MNIST,
        nn,
        optim,
        plt,
        random,
        torch,
        transforms,
    )


@app.cell
def _(mo):
    mo.md(r"""### Dataset Creation""")
    return


@app.cell
def _(MNIST, transforms):
    mnist_train = MNIST(root="./data", train=True, download=True)
    mnist_test = MNIST(root="./data", train=False, download=True)

    transform = transforms.Compose([transforms.ToTensor()])
    return mnist_test, mnist_train, transform


@app.cell
def _(Dataset, random, torch):
    class SiameseDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            imgA, labelA = self.data[index]

            same_class_flag = random.randint(0, 1)  # Pair with same class?

            if same_class_flag:  # yes, pair with same class
                labelB = -1
                while labelB != labelA:
                    imgB, labelB = random.choice(self.data)

            else:  # no, pair with different class
                labelB = labelA
                while labelB == labelA:
                    imgB, labelB = random.choice(self.data)

            if self.transform:
                imgA = self.transform(imgA)
                imgB = self.transform(imgB)

            pair_label = torch.tensor(
                [(labelA != labelB)], dtype=torch.float32
            )  # 0 if same label, else 1

            return imgA, imgB, pair_label
    return (SiameseDataset,)


@app.cell
def _(SiameseDataset, mnist_test, mnist_train, transform):
    siamese_train = SiameseDataset(mnist_train, transform)
    siamese_test = SiameseDataset(mnist_test, transform)
    return siamese_test, siamese_train


@app.cell
def _(plt, siamese_train):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(siamese_train[0][0][-1], cmap="gist_earth")
    axes[0].set_title("Image A")
    axes[0].axis("off")
    axes[1].imshow(siamese_train[0][1][-1], cmap="magma")
    axes[1].set_title("Image B")
    axes[1].axis("off")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""### Siamese Network""")
    return


@app.cell
def _(nn):
    class SiameseNetwork(nn.Module):
        def __init__(self):
            super(SiameseNetwork, self).__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
            )

            self.fc = nn.Sequential(
                nn.Linear(256 * 4 * 4, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
            )

        def forward_once(self, x):
            output = self.cnn(x)
            output = output.view(output.size()[0], -1)
            output = self.fc(output)
            return output

        def forward(self, inputA, inputB):
            outputA = self.forward_once(inputA)
            outputB = self.forward_once(inputB)
            return outputA, outputB
    return (SiameseNetwork,)


@app.cell
def _(mo):
    mo.md(r"""### Contrastive Loss""")
    return


@app.cell
def _(F, torch):
    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, outputA, outputB, y):
            euclidean_distance = F.pairwise_distance(
                outputA, outputB, keepdim=True
            )
            same_class_loss = (1 - y) * (euclidean_distance**2)
            diff_class_loss = (y) * (
                torch.clamp(self.margin - euclidean_distance, min=0.0) ** 2
            )

            # NOTE: Mean across every dimension of the loss vector
            return torch.mean(same_class_loss + diff_class_loss)
    return (ContrastiveLoss,)


@app.cell
def _(mo):
    mo.md(r"""### Model Training""")
    return


@app.cell
def _(ContrastiveLoss, DataLoader, SiameseNetwork, optim, siamese_train):
    train_dataloader = DataLoader(
        siamese_train, shuffle=True, num_workers=8, batch_size=64
    )
    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, model, optimizer, train_dataloader


@app.cell
def _(criterion, model, optimizer, train_dataloader):
    for epoch in range(10):
        total_loss = 0

        for imgA, imgB, label in train_dataloader:
            imgA, imgB, label = imgA.cuda(), imgB.cuda(), label.cuda()
            optimizer.zero_grad()
            outputA, outputB = model.forward(imgA, imgB)
            loss_contrastive = criterion(outputA, outputB, label)
            loss_contrastive.backward()

            total_loss += loss_contrastive.item()
            optimizer.step()

        print(f"Epoch {epoch}; Loss {total_loss}")
    return


@app.cell
def _(model, torch):
    model_path = "siamese_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to `{model_path}`")
    return


@app.cell
def _(mo):
    mo.md(r"""### Model Inferencing""")
    return


@app.cell
def _(F, SiameseNetwork, torch):
    def load_model(model_path, device):
        inference_model = SiameseNetwork()
        inference_model.load_state_dict(torch.load(model_path))
        return inference_model.to(device)


    def get_similarity_score(image1, image2, model, device):
        # Ensure model is in evaluation mode
        model.eval()
        with torch.no_grad():  # No need to calculate gradients during inference
            # Move images to the correct device
            img1 = image1.to(device)
            img2 = image2.to(device)

            # Get embeddings from the model
            output1 = model.forward_once(img1.unsqueeze(0))  # Add batch dimension
            output2 = model.forward_once(img2.unsqueeze(0))  # Add batch dimension

            # Calculate Euclidean distance
            euclidean_distance = F.pairwise_distance(output1, output2)
            return torch.exp(-euclidean_distance).cpu().numpy()[0]
    return get_similarity_score, load_model


@app.cell
def _(siamese_test):
    len(siamese_test)
    return


@app.cell
def _(plt, random, siamese_test, siamese_train):
    def get_random_image_pairs():
        same_class_pair = None
        diff_class_pair = None

        # Find a same-class pair (label == 0)
        while same_class_pair is None:
            idx = random.randint(0, len(siamese_test) - 1)  # Corrected range
            imgA_s, imgB_s, label_s = siamese_test[idx]
            if label_s.item() == 0:
                same_class_pair = (imgA_s, imgB_s)

        # Find a different-class pair (label == 1)
        while diff_class_pair is None:
            idx = random.randint(0, len(siamese_test) - 1)  # Corrected range
            imgA_d, imgB_d, label_d = siamese_test[idx]
            if label_d.item() == 1:
                diff_class_pair = (imgA_d, imgB_d)

        return same_class_pair, diff_class_pair


    def plot_imgs(imgA, imgB, similarity_score, label):
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(siamese_train[0][0][-1], cmap="gist_earth")
        # axes[0].set_title("Image A")
        axes[0].axis("off")
        axes[1].imshow(siamese_train[0][1][-1], cmap="magma")
        # axes[1].set_title("Image B")
        axes[1].axis("off")
        type = "same" if label == 0 else "diff"
        plt.suptitle(
            f"{similarity_score:0.02f} similarity Score Between {type} images",
            y=1.05,
        )
        plt.tight_layout()
        plt.gca()
        plt.show()
    return get_random_image_pairs, plot_imgs


@app.cell
def _(
    get_random_image_pairs,
    get_similarity_score,
    load_model,
    plot_imgs,
    torch,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model = load_model(model_path="siamese_model.pth", device=device)

    for i in range(3):
        sp, dp = get_random_image_pairs()
        score_sp = get_similarity_score(sp[0], sp[1], inference_model, device)
        plot_imgs(sp[0], sp[1], score_sp, 0)

        score_dp = get_similarity_score(dp[0], dp[1], inference_model, device)
        plot_imgs(dp[0], dp[1], score_dp, 1)
    return


if __name__ == "__main__":
    app.run()
