import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
import wandb
import pandas as pd
import sys
import argparse
from titanic_dataset import get_preprocessed_dataset

def get_train_data(batch_size):
    train_dataset, validation_dataset, _ = get_preprocessed_dataset()
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset))
    return train_data_loader, validation_data_loader

def get_test_data():
    _, _, test_dataset = get_preprocessed_dataset()
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
    return test_data_loader

class MyModel(nn.Module):
    def __init__(self, n_input, n_output, activation_function):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),  # 첫 번째 은닉층
            activation_function(),  # 활성화 함수
            nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),  # 두 번째 은닉층
            activation_function(),  # 활성화 함수
            nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),  # 출력층
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_model_and_optimizer(activation_fn):
    model = MyModel(n_input=11, n_output=2, activation_function=activation_fn)
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
    return model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
    n_epochs = wandb.config.epochs
    loss_fn = nn.CrossEntropyLoss()
    best_validation_accuracy = 0.0
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        loss_train = 0.0
        correct_train = 0
        num_trains = 0

        # 학습 데이터 처리
        for batch in train_data_loader:
            input = batch['input']
            target = batch['target']

            output_train = model(input)
            loss = loss_fn(output_train, target)
            loss_train += loss.item()

            prediction = torch.argmax(output_train, dim=1)
            correct_train += (prediction == target).sum().item()
            num_trains += target.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        loss_validation = 0.0
        correct_validation = 0
        num_validations = 0

        with torch.no_grad():
            for batch in validation_data_loader:
                input = batch['input']
                target = batch['target']

                output_validation = model(input)
                loss = loss_fn(output_validation, target)
                loss_validation += loss.item()

                prediction = torch.argmax(output_validation, dim=1)
                correct_validation += (prediction == target).sum().item()
                num_validations += target.size(0)

        validation_accuracy = correct_validation / num_validations

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = model.state_dict()

        wandb.log({
            "Epoch": epoch,
            "Training Loss": loss_train / num_trains,
            "Training Accuracy": correct_train / num_trains,
            "Validation Loss": loss_validation / num_validations,
            "Validation Accuracy": validation_accuracy
        })

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | "
                  f"Train Loss: {loss_train / num_trains:.4f} | "
                  f"Train Acc: {correct_train / num_trains:.4f} | "
                  f"Val Loss: {loss_validation / num_validations:.4f} | "
                  f"Val Acc: {validation_accuracy:.4f}")

    return best_validation_accuracy, best_model_state


def test_model(model, test_data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_data_loader:
            input = batch['input']
            output = model(input)
            prediction = torch.argmax(output, dim=1)
            predictions.extend(prediction.tolist())

    return predictions


def save_submission(predictions, filename="submission.csv"):
    submission = pd.DataFrame({
        "PassengerId": range(892, 892 + len(predictions)),
        "Survived": predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file has been successfully saved as '{filename}'")


def main(args):
    # 현재 시간을 문자열로 저장
    timestamp = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    # 실험 설정 내용
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 1e-3,
        'n_hidden_unit_list': [30, 30],
    }

    # W&B 설정 초기화
    wandb.init(
        mode="online",
        project="titanic_model_training_with_ReLU",
        notes="Titanic dataset classification with ReLU",
        tags=["titanic", "classification", "ReLU"],
        name=timestamp,
        config=config
    )

    # 학습 및 검증 데이터 로드
    train_data_loader, validation_data_loader = get_train_data(wandb.config.batch_size)
    # 모델 및 옵티마이저 초기화
    model, optimizer = get_model_and_optimizer(nn.ReLU)

    # Relu 활성화 함수 사용
    relu_validation_accuracy, _ = training_loop(
        model=model,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader
    )
    wandb.finish()

    activation = {
        "PReLU": nn.PReLU,
        "ELU": nn.ELU,
        "Leaky ReLU": nn.LeakyReLU
    }

    optimal_activation = "ReLU"
    highest_val_accuracy = relu_validation_accuracy
    optimal_model_state = None

    for activation_name, activation_fn in activation.items():
        print(f"Starting experiment with {activation_name}")
        wandb.init(
            mode="online",
            project=f"titanic_model_training_with_{activation_name}",
            notes=f"Titanic dataset classification with {activation_name}",
            tags=["titanic", "classification", activation_name],
            name=f"{activation_name}_{timestamp}",
            config=config
        )
        print(f"WandB is initialized for {activation_name}")

        model, optimizer = get_model_and_optimizer(activation_fn)

        validation_accuracy, model_state = training_loop(
            model=model,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            validation_data_loader=validation_data_loader
        )

        print(f"{activation_name} Validation Accuracy: {validation_accuracy:.4f}")

        if validation_accuracy > highest_val_accuracy:
            highest_val_accuracy = validation_accuracy
            optimal_activation = activation_name
            optimal_model_state = model_state

        wandb.finish()

    print(
        f"The best activation function is {optimal_activation} with a validation accuracy of {highest_val_accuracy:.4f}")

    print(f"Testing the best model with {optimal_activation}")
    model.load_state_dict(optimal_model_state)
    test_data_loader = get_test_data()
    test_predictions = test_model(model, test_data_loader)
    save_submission(test_predictions)


if __name__ == "__main__":
    if 'ipykernel_launcher' in sys.argv[0]:
        sys.argv = [sys.argv[0]]
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False, help="True or False"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="Batch size (int, default: 16)"
    )

    parser.add_argument(
        "-e", "--epochs", type=int, default=1000, help="Number of training epochs (int, default:1000)"
    )

    args = parser.parse_args()

    main(args)
