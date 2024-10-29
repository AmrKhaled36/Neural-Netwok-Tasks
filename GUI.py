import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import matplotlib.pyplot as plt
import numpy as np
import perceptron as per

class SignalApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Task 1 NN")

        for i in range(5):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.root.grid_columnconfigure(i, weight=1)

        self.features = ["Gender", "Body mass", "Beak length", "Beak depth","Fin length"]
        self.classes = ["A", "B", "C"]
        ##First row of combo boxes.
        self.feature1 = ttk.Combobox(self.root, values=self.features, state="readonly")
        self.feature1.grid(row=0, column=0, padx=10, pady=10)
        self.feature1.set("Select feature 1")
        self.feature1.bind("<<ComboboxSelected>>", self.selected_features_classes)

        self.feature2 = ttk.Combobox(self.root, values=self.features, state="readonly")
        self.feature2.grid(row=0, column=1, padx=10, pady=10)
        self.feature2.set("Select feature 2")
        self.feature2.bind("<<ComboboxSelected>>", self.selected_features_classes)

        ##Second row of combo boxes.
        self.class1 = ttk.Combobox(self.root, values=self.classes, state="readonly")
        self.class1.grid(row=1, column=0, padx=10, pady=10)
        self.class1.set("Select class 1")
        self.class1.bind("<<ComboboxSelected>>", self.selected_features_classes)

        self.class2 = ttk.Combobox(self.root, values=self.classes, state="readonly")
        self.class2.grid(row=1, column=1, padx=10, pady=10)
        self.class2.set("Select class 2")
        self.class2.bind("<<ComboboxSelected>>", self.selected_features_classes)

        #Third row text boxes
        self.learning_rate_label = tk.Label(self.root, text="Learning rate:")
        self.learning_rate_label.grid(row=2, column=0, padx=10, pady=10)

        self.epochs_label = tk.Label(self.root, text="Epochs:")
        self.epochs_label.grid(row=2, column=1, padx=10, pady=10)

        self.mse_label = tk.Label(self.root, text="MSE threshold:")
        self.mse_label.grid(row=2, column=2, padx=10, pady=10)

        ##fourth row entries
        self.learning_rate_entry = tk.Entry(self.root)
        self.learning_rate_entry.grid(row=3, column=0, padx=10, pady=10)

        self.epochs_entry = tk.Entry(self.root)
        self.epochs_entry.grid(row=3, column=1, padx=10, pady=10)

        self.mse_entry = tk.Entry(self.root)
        self.mse_entry.grid(row=3, column=2, padx=10, pady=10)

        ##fifth row
        self.bias_var = tk.BooleanVar()
        self.bias_check = tk.Checkbutton(self.root, text="Bias", variable=self.bias_var, command=self.bias_check_clicked)
        self.bias_check.grid(row=4, column=0, padx=10, pady=10)
        self.bias_entry = tk.Entry(self.root)
        self.bias_entry.config(state="disabled")
        self.bias_entry.grid(row=4, column=1, padx=10, pady=10)

        #ٌRadio buttons
        self.radio_var = tk.StringVar()
        self.radio_var.set("perceptron")

        self.radio_percep = tk.Radiobutton(self.root, text="Perceptron", variable=self.radio_var, value="perceptron")
        self.radio_percep.grid(row=0, column=2, padx=10, pady=10)

        self.radio_adal = tk.Radiobutton(self.root, text="Adaline", variable=self.radio_var, value="adaline")
        self.radio_adal.grid(row=1, column=2, padx=10, pady=10)

    def selected_features_classes(self,event):
        if self.feature1.get() == self.feature2.get():
            messagebox.showerror("Error", "Please select different features")
            self.feature1.set("Select feature 1")
            self.feature2.set("Select feature 2")
        if self.class1.get() == self.class2.get():
            messagebox.showerror("Error", "Please select different classes")
            self.class1.set("Select class 1")
            self.class2.set("Select class 2")

    def bias_check_clicked(self):
        if self.bias_var.get():
            self.bias_entry.config(state="normal")
        else:
            self.bias_entry.config(state="disabled")

