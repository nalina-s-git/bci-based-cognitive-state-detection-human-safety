import matplotlib.pyplot as plt

accuracy = 70.11  # your result

plt.figure(figsize=(5,4))
plt.bar(['SVM Model'], [accuracy])
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.text(0, accuracy + 1, f"{accuracy:.2f}%", ha='center')
plt.show()