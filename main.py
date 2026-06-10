# import csv
# import os
# import trainer

# LOSSES_CSV = "losses.csv"






#legacy
#main for evolutionary agent
"""
def append_run_to_csv(losses: list[float], params: list[int]) -> None:
    "Append one training run as a new row. Creates the file with headers on first call."
    layer_str = " -> ".join(str(w) for w in params)
    arch = f"1 -> {layer_str} -> 1"
    num_epochs = len(losses)

    needs_header = not os.path.exists(LOSSES_CSV)
    with open(LOSSES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            header = ["architecture", "lr", "epochs", "batch_size"] + [f"epoch_{i+1}" for i in range(num_epochs)]
            writer.writerow(header)
        writer.writerow([arch, trainer.LR, trainer.EPOCHS, trainer.BATCH_SIZE] + losses)


params = [32, 32]
losses = trainer.train(params)

depth = len(params)
layer_str = " -> ".join(str(w) for w in params)
architecture = f"1 -> {layer_str} -> 1"

description = (
    f"=== Run Summary ===\n"
    f"Task:         sin(x) approximation\n"
    f"Architecture: MLP  {architecture}\n"
    f"Depth:        {depth} hidden layer(s)\n"
    f"Widths:       {params}\n"
    f"\n"
    f"--- Training Hyperparameters ---\n"
    f"Optimizer:    Adam\n"
    f"LR:           {trainer.LR}\n"
    f"Epochs:       {trainer.EPOCHS}\n"
    f"Batch size:   {trainer.BATCH_SIZE}\n"
    f"\n"
    f"--- Results ---\n"
    f"Final loss:   {losses[-1]:.6f}\n"
    f"Best loss:    {min(losses):.6f}  (epoch {losses.index(min(losses)) + 1})\n"
    f"All losses: {', '.join(f'{loss:.6f}' for loss in losses)}"
)

with open("results.txt", "w") as f:
    f.write(description)

append_run_to_csv(losses, params)

print(description)
print("done")
"""
