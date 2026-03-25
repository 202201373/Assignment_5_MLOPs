import argparse
import pathlib
import time
import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """A small but effective CNN for MNIST digit classification."""

    def __init__(self, dropout_rate: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def get_dataloaders(batch_size: int, data_dir: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_experiment(
    learning_rate: float = 0.001,
    batch_size: int      = 64,
    epochs: int          = 5,
    dropout_rate: float  = 0.25,
    optimizer_name: str  = "adam",
    run_name: str        = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Device        : {device}")
    print(f"  learning_rate : {learning_rate}")
    print(f"  batch_size    : {batch_size}")
    print(f"  epochs        : {epochs}")
    print(f"  dropout_rate  : {dropout_rate}")
    print(f"  optimizer     : {optimizer_name}")
    print(f"{'='*60}\n")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlruns_dir = pathlib.Path(__file__).parent / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(mlruns_dir.as_uri())

    mlflow.set_experiment("MNIST-Pipeline")

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.set_tag("student_name", "Reem Ehab")
        mlflow.set_tag("model",        "SimpleCNN")
        mlflow.set_tag("dataset",      "MNIST")
        mlflow.set_tag("device",       str(device))

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size",    batch_size)
        mlflow.log_param("epochs",        epochs)
        mlflow.log_param("dropout_rate",  dropout_rate)
        mlflow.log_param("optimizer",     optimizer_name)

        model     = SimpleCNN(dropout_rate=dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()

        if optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_name.lower() == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loader, test_loader = get_dataloaders(batch_size)

        start_time = time.time()
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)

            mlflow.log_metric("train_loss",     train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc,  step=epoch)
            mlflow.log_metric("val_loss",       val_loss,   step=epoch)
            mlflow.log_metric("val_accuracy",   val_acc,    step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(
                f"  Epoch {epoch:>2}/{epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

        elapsed = time.time() - start_time
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.log_metric("training_time_sec", elapsed)
        mlflow.log_metric("accuracy", best_val_acc)

        mlflow.pytorch.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        with open("model_info.txt", "w") as f:
            f.write(run_id)
        print(f"\n  ✓ Run ID saved to model_info.txt: {run_id}")

        print(f"  ✓ Run complete. Best val_accuracy = {best_val_acc:.4f}  ({elapsed:.1f}s)\n")
        return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow-instrumented CNN trainer (MNIST)")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--dropout_rate",  type=float, default=0.25)
    parser.add_argument("--optimizer",     type=str,   default="adam",
                        choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--run_name",      type=str,   default=None)
    args = parser.parse_args()

    run_experiment(
        learning_rate = args.learning_rate,
        batch_size    = args.batch_size,
        epochs        = args.epochs,
        dropout_rate  = args.dropout_rate,
        optimizer_name= args.optimizer,
        run_name      = args.run_name,
    )
