from src.DL.aerial_data_loader import get_data_loaders
from src.DL.aerial_model_factory import get_resnet18, get_mobilenetv3_small
from src.DL.aerial_training_utils import train_model, evaluate_model
from src.DL.aerial_occlusion_test import evaluate_with_occlusion
from src.DL.aerial_scorecam import get_scorecam_heatmap
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    """
    Main function to run the aerial scene classification experiment, including data loading, model training,
    Score-CAM visualization, and occlusion robustness testing.
    """
    # Experiment configuration
    data_dir = '../Aerial_Landscapes'
    batch_size = 64
    epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and create DataLoaders
    train_loader1, train_loader2, test_loader = get_data_loaders(data_dir, batch_size, train_ratio=0.8, sample_ratio=1)
    print("Data loaded successfully.")
    
    # Get class names
    class_names = test_loader.dataset.dataset.classes
    
    # Initialize results list
    all_results = []
    
    # Model 1: ResNet-18, without CutMix
    model1 = get_resnet18(num_classes=15).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    print("Training Model 1 (ResNet-18, No CutMix)...")
    best_acc1, training_time1, train_loss1 = train_model(model1, train_loader1, test_loader, criterion, optimizer1, device, epochs=epochs, use_cutmix=False)
    acc1_base, f1_1_base, prec1_base, rec1_base, iou1_base, _, _ = evaluate_model(model1, test_loader, device, class_names=class_names, scenario="Model 1 Baseline (No Occlusion)", train_loss=train_loss1)
    print(f"Model 1 Baseline: Accuracy={acc1_base:.2f}%, F1={f1_1_base:.2f}, Precision={prec1_base:.2f}, Recall={rec1_base:.2f}, IoU={iou1_base:.3f}, Train loss={train_loss1:.3f}")

    # Model 2: ResNet-18, with CutMix
    model2 = get_resnet18(num_classes=15).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    print("Training Model 2 (ResNet-18, With CutMix)...")
    best_acc2, training_time2, train_loss2 = train_model(model2, train_loader2, test_loader, criterion, optimizer2, device, epochs=epochs, use_cutmix=True)
    acc2_base, f1_2_base, prec2_base, rec2_base, iou2_base, _, _ = evaluate_model(model2, test_loader, device, class_names=class_names, scenario="Model 2 Baseline (No Occlusion)", train_loss=train_loss2)
    print(f"Model 2 Baseline: Accuracy={acc2_base:.2f}%, F1={f1_2_base:.2f}, Precision={prec2_base:.2f}, Recall={rec2_base:.2f}, IoU={iou2_base:.3f}, Train loss={train_loss2:.3f}")

    # Model 3: MobileNetV3-Small, without CutMix
    model3 = get_mobilenetv3_small(num_classes=15).to(device)
    optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
    print("Training Model 3 (MobileNetV3-Small, No CutMix)...")
    best_acc3, training_time3, train_loss3 = train_model(model3, train_loader1, test_loader, criterion, optimizer3, device, epochs=epochs, use_cutmix=False)
    acc3_base, f1_3_base, prec3_base, rec3_base, iou3_base, _, _ = evaluate_model(model3, test_loader, device, class_names=class_names, scenario="Model 3 Baseline (No Occlusion)", train_loss=train_loss3)
    print(f"Model 3 Baseline: Accuracy={acc3_base:.2f}%, F1={f1_3_base:.2f}, Precision={prec3_base:.2f}, Recall={rec3_base:.2f}, IoU={iou3_base:.3f}, Train loss={train_loss3:.3f}")

    # Model 4: MobileNetV3-Small, with CutMix
    model4 = get_mobilenetv3_small(num_classes=15).to(device)
    optimizer4 = optim.Adam(model4.parameters(), lr=0.001)
    print("Training Model 4 (MobileNetV3-Small, With CutMix)...")
    best_acc4, training_time4, train_loss4 = train_model(model4, train_loader2, test_loader, criterion, optimizer4, device, epochs=epochs, use_cutmix=True)
    acc4_base, f1_4_base, prec4_base, rec4_base, iou4_base, _, _ = evaluate_model(model4, test_loader, device, class_names=class_names, scenario="Model 4 Baseline (No Occlusion)", train_loss=train_loss4)
    print(f"Model 4 Baseline: Accuracy={acc4_base:.2f}%, F1={f1_4_base:.2f}, Precision={prec4_base:.2f}, Recall={rec4_base:.2f}, IoU={iou4_base:.3f}, Train loss={train_loss4:.3f}")

    # Occlusion robustness test
    occlusion_sizes = [32, 64, 96]
    print("\nEvaluating occlusion robustness...")
    for size in occlusion_sizes:
        print(f"\nEvaluating occlusion size {size}px for Model 1...")
        acc1_occ, f1_1_occ, prec1_occ, rec1_occ, iou1_occ = evaluate_with_occlusion(model1, test_loader, size, device, class_names, train_loss1)
        print(f"Model 1 - Occlusion {size}px (Before vs After):")
        print(f"Accuracy: {acc1_base:.2f}% (Before) -> {acc1_occ:.2f}% (After)")
        print(f"F1 Score: {f1_1_base:.2f} (Before) -> {f1_1_occ:.2f} (After)")
        print(f"Precision: {prec1_base:.2f} (Before) -> {prec1_occ:.2f} (After)")
        print(f"Recall: {rec1_base:.2f} (Before) -> {rec1_occ:.2f} (After)")
        print(f"IoU: {iou1_base:.3f} (Before) -> {iou1_occ:.3f} (After)")
        print(f"Train loss: {train_loss1:.3f}")
        all_results.append({
            "Model": "Model1 (ResNet-18, No CutMix)",
            "Best Test Accuracy": f"{best_acc1:.2f}%",
            "Occlusion Size": size,
            "Occlusion Accuracy": f"{acc1_occ:.2f}%",
            "Occlusion F1": f"{f1_1_occ:.3f}",
            "Occlusion Precision": f"{prec1_occ:.3f}",
            "Occlusion Recall": f"{rec1_occ:.3f}",
            "Occlusion IoU": f"{iou1_occ:.3f}",
            "Train loss": f"{train_loss1:.3f}",
            "Baseline Accuracy": f"{acc1_base:.2f}%",
            "Baseline F1": f"{f1_1_base:.3f}",
            "Baseline Precision": f"{prec1_base:.3f}",
            "Baseline Recall": f"{rec1_base:.3f}",
            "Baseline IoU": f"{iou1_base:.3f}",
            "TrainTime(s)": f"{training_time1:.1f}"
        })
        
        print(f"\nEvaluating occlusion size {size}px for Model 2...")
        acc2_occ, f1_2_occ, prec2_occ, rec2_occ, iou2_occ = evaluate_with_occlusion(model2, test_loader, size, device, class_names, train_loss2)
        print(f"Model 2 - Occlusion {size}px (Before vs After):")
        print(f"Accuracy: {acc2_base:.2f}% (Before) -> {acc2_occ:.2f}% (After)")
        print(f"F1 Score: {f1_2_base:.2f} (Before) -> {f1_2_occ:.2f} (After)")
        print(f"Precision: {prec2_base:.2f} (Before) -> {prec2_occ:.2f} (After)")
        print(f"Recall: {rec2_base:.2f} (Before) -> {rec2_occ:.2f} (After)")
        print(f"IoU: {iou2_base:.3f} (Before) -> {iou2_occ:.3f} (After)")
        print(f"Train loss: {train_loss2:.3f}")
        all_results.append({
            "Model": "Model2 (ResNet-18, With CutMix)",
            "Best Test Accuracy": f"{best_acc2:.2f}%",
            "Occlusion Size": size,
            "Occlusion Accuracy": f"{acc2_occ:.2f}%",
            "Occlusion F1": f"{f1_2_occ:.3f}",
            "Occlusion Precision": f"{prec2_occ:.3f}",
            "Occlusion Recall": f"{rec2_occ:.3f}",
            "Occlusion IoU": f"{iou2_occ:.3f}",
            "Train loss": f"{train_loss2:.3f}",
            "Baseline Accuracy": f"{acc2_base:.2f}%",
            "Baseline F1": f"{f1_2_base:.3f}",
            "Baseline Precision": f"{prec2_base:.3f}",
            "Baseline Recall": f"{rec2_base:.3f}",
            "Baseline IoU": f"{iou2_base:.3f}",
            "TrainTime(s)": f"{training_time2:.1f}"
        })

        print(f"\nEvaluating occlusion size {size}px for Model 3...")
        acc3_occ, f1_3_occ, prec3_occ, rec3_occ, iou3_occ = evaluate_with_occlusion(model3, test_loader, size, device, class_names, train_loss3)
        print(f"Model 3 - Occlusion {size}px (Before vs After):")
        print(f"Accuracy: {acc3_base:.2f}% (Before) -> {acc3_occ:.2f}% (After)")
        print(f"F1 Score: {f1_3_base:.2f} (Before) -> {f1_3_occ:.2f} (After)")
        print(f"Precision: {prec3_base:.2f} (Before) -> {prec3_occ:.2f} (After)")
        print(f"Recall: {rec3_base:.2f} (Before) -> {rec3_occ:.2f} (After)")
        print(f"IoU: {iou3_base:.3f} (Before) -> {iou3_occ:.3f} (After)")
        print(f"Train loss: {train_loss3:.3f}")
        all_results.append({
            "Model": "Model3 (MobileNetV3-Small, No CutMix)",
            "Best Test Accuracy": f"{best_acc3:.2f}%",
            "Occlusion Size": size,
            "Occlusion Accuracy": f"{acc3_occ:.2f}%",
            "Occlusion F1": f"{f1_3_occ:.3f}",
            "Occlusion Precision": f"{prec3_occ:.3f}",
            "Occlusion Recall": f"{rec3_occ:.3f}",
            "Occlusion IoU": f"{iou3_occ:.3f}",
            "Train loss": f"{train_loss3:.3f}",
            "Baseline Accuracy": f"{acc3_base:.2f}%",
            "Baseline F1": f"{f1_3_base:.3f}",
            "Baseline Precision": f"{prec3_base:.3f}",
            "Baseline Recall": f"{rec3_base:.3f}",
            "Baseline IoU": f"{iou3_base:.3f}",
            "TrainTime(s)": f"{training_time3:.1f}"
        })

        print(f"\nEvaluating occlusion size {size}px for Model 4...")
        acc4_occ, f1_4_occ, prec4_occ, rec4_occ, iou4_occ = evaluate_with_occlusion(model4, test_loader, size, device, class_names, train_loss4)
        print(f"Model 4 - Occlusion {size}px (Before vs After):")
        print(f"Accuracy: {acc4_base:.2f}% (Before) -> {acc4_occ:.2f}% (After)")
        print(f"F1 Score: {f1_4_base:.2f} (Before) -> {f1_4_occ:.2f} (After)")
        print(f"Precision: {prec4_base:.2f} (Before) -> {prec4_occ:.2f} (After)")
        print(f"Recall: {rec4_base:.2f} (Before) -> {rec4_occ:.2f} (After)")
        print(f"IoU: {iou4_base:.3f} (Before) -> {iou4_occ:.3f} (After)")
        print(f"Train loss: {train_loss4:.3f}")
        all_results.append({
            "Model": "Model4 (MobileNetV3-Small, With CutMix)",
            "Best Test Accuracy": f"{best_acc4:.2f}%",
            "Occlusion Size": size,
            "Occlusion Accuracy": f"{acc4_occ:.2f}%",
            "Occlusion F1": f"{f1_4_occ:.3f}",
            "Occlusion Precision": f"{prec4_occ:.3f}",
            "Occlusion Recall": f"{rec4_occ:.3f}",
            "Occlusion IoU": f"{iou4_occ:.3f}",
            "Train loss": f"{train_loss4:.3f}",
            "Baseline Accuracy": f"{acc4_base:.2f}%",
            "Baseline F1": f"{f1_4_base:.3f}",
            "Baseline Precision": f"{prec4_base:.3f}",
            "Baseline Recall": f"{rec4_base:.3f}",
            "Baseline IoU": f"{iou4_base:.3f}",
            "TrainTime(s)": f"{training_time4:.1f}"
        })
    
    print("\nGenerating Score-CAM heatmap...")
    sample_image, sample_label = next(iter(test_loader))
    sample_image = sample_image[0].unsqueeze(0).to(device)
    target_layer = model1.layer4[-1]
    heatmap = get_scorecam_heatmap(model1, sample_image, target_layer, sample_label[0].item())
    print(f"Score-CAM heatmap generated, shape: {heatmap.shape}")

if __name__ == "__main__":
    main()