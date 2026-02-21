import re
import wandb

# === CONFIGURATION ===
LOG_FILE = "/home/yli11/scratch/Hafeez_thesis/Can3Tok/logs/bala_10k_geometric.out"
WANDB_PROJECT = "can3tok-retroactive"
WANDB_ENTITY = None  # Set to your wandb username or team if needed

# === REGEX PATTERNS ===
train_pattern = re.compile(
    r"Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Recon: ([\d.]+) \| KL: ([\d.]+) \| Semantic: ([\d.]+)"
)
val_pattern = re.compile(
    r"VALIDATION \(Epoch (\d+)\).*?L2 Error: ([\d.]+) ± ([\d.]+).*?Failure Rate: ([\d.]+)%",
    re.DOTALL
)

# === PARSE LOG FILE ===
train_metrics = []
val_metrics = []

with open(LOG_FILE, "r") as f:
    log = f.read()

# Parse training metrics
for match in train_pattern.finditer(log):
    epoch, loss, recon, kl, semantic = match.groups()
    train_metrics.append({
        "epoch": int(epoch),
        "loss": float(loss),
        "recon": float(recon),
        "kl": float(kl),
        "semantic": float(semantic),
    })

# Parse validation metrics
for match in val_pattern.finditer(log):
    epoch, l2, l2_std, fail_rate = match.groups()
    val_metrics.append({
        "epoch": int(epoch),
        "val_l2": float(l2),
        "val_l2_std": float(l2_std),
        "val_failure_rate": float(fail_rate),
    })

# === LOG TO WANDB ===
wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name="retroactive-log-upload")

for tm in train_metrics:
    wandb.log({
        "epoch": tm["epoch"],
        "loss": tm["loss"],
        "recon": tm["recon"],
        "kl": tm["kl"],
        "semantic": tm["semantic"],
    }, step=tm["epoch"])

for vm in val_metrics:
    wandb.log({
        "epoch": vm["epoch"],
        "val_l2": vm["val_l2"],
        "val_l2_std": vm["val_l2_std"],
        "val_failure_rate": vm["val_failure_rate"],
    }, step=vm["epoch"])

wandb.finish()
print("Done uploading metrics to wandb.")