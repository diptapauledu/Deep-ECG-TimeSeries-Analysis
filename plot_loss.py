import matplotlib.pyplot as plt

# Example loss values (replace with your actual logged losses)
losses = [
    0.092, 0.081, 0.073, 0.066, 0.060,
    0.055, 0.051, 0.048, 0.045, 0.043
]

epochs = range(1, len(losses) + 1)

plt.figure(figsize=(6, 4))
plt.plot(epochs, losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Convergence")
plt.grid(True)
plt.tight_layout()

plt.savefig("results/training_loss.png", dpi=300)
plt.show()
