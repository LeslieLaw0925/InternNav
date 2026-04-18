import re
import matplotlib.pyplot as plt
import numpy as np


log_file = "log_20260402144406_3842671.log"

traj_uncertainties = []
img_uncertainties = []
endpoint_vars = []

pattern = re.compile(
    r"Trajectory latent uncertainty: ([\d\.]+), "
    r"Image token uncertainty: ([\d\.]+), "
    r"Endpoint variance: ([\d\.]+)"
)

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            traj_uncertainties.append(float(match.group(1)))
            img_uncertainties.append(float(match.group(2)))
            endpoint_vars.append(float(match.group(3)))

# traj_uncertainties, img_uncertainties, endpoint_vars = np.array(traj_uncertainties), np.array(img_uncertainties), np.array(endpoint_vars)
print("Extracted:", len(traj_uncertainties), "entries")

data = np.array([
    traj_uncertainties,
    img_uncertainties,
    endpoint_vars
])

corr = np.corrcoef(data)
print(corr)


plt.figure()
plt.plot(traj_uncertainties, label="traj_unc")
# plt.plot(img_uncertainties * 10, label="img_unc")
plt.plot(endpoint_vars, label="endpoint_var")
plt.legend()
plt.title("Uncertainty over time")
plt.savefig("uncertainties.png")