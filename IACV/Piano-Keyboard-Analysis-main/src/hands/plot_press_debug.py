import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "outputs/taskB/b2_pressed_per_finger_clean.csv"

bend_col = "right_thumb_bend"
press_col = "right_thumb_pressed"

df = pd.read_csv(CSV)

bend = df[bend_col].values.astype(float)
press = df[press_col].values.astype(float)
x = np.arange(len(bend))

plt.figure(figsize=(10, 6))

# bend curve
plt.plot(x, bend, color="tab:blue", label=bend_col)

# press markers: show as dots ON the bend curve
press_idx = np.where(press == 1)[0]
plt.scatter(press_idx, bend[press_idx], color="tab:orange", s=40, label=press_col)

plt.title("Bend + Press markers (clean)")
plt.xlabel("frame")
plt.legend()
plt.tight_layout()
plt.show()