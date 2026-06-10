from matplotlib import pyplot
import csv

LOSSES_CSV = "losses.csv"

with open(LOSSES_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        arch = row["architecture"]
        num_epochs = sum(1 for k in row if k.startswith("epoch_"))
        all_losses = [float(row[f"epoch_{e}"]) for e in range(1, num_epochs + 1)]
        window = 250
        x_points = [0]
        y_points = [all_losses[0]]
        for start in range(0, num_epochs, window):
            chunk = all_losses[start:start + window]
            x_points.append(start + window)
            y_points.append(sum(chunk) / len(chunk))
        pyplot.plot(x_points, y_points, label=arch)

pyplot.axhline(y=0.001, color="red", linewidth=1.5, label="benchmark (0.001)")
pyplot.yscale("log")
pyplot.xlabel("Epoch")
pyplot.ylabel("MSE Loss (log scale)")
pyplot.title("Training Loss by Architecture")
pyplot.legend()
pyplot.grid(True, which="both", linestyle="--", alpha=0.5)

pyplot.show()
